import os
import argparse

from lib.cli import add_common_args, finalize_args
from lib.paths import resolve_dataset_paths
from lib.paths import resolve_experiment_dirs

from lib.evaluate import *

parser = argparse.ArgumentParser(description='PyTorch Prediction Model on Time-series Dataset')
add_common_args(parser)
args = finalize_args(parser.parse_args())

# Resolve experiment directories
exp = resolve_experiment_dirs(getattr(args, 'log_dir', 'expe'))
args.run_id = exp.run_id
args.log_dir = exp.root
args.log_dir_log = exp.log_dir
args.log_dir_pth = exp.pth_dir
args.log_dir_pdf = exp.pdf_dir

# Provide a reasonable default for NPZ mode (SMD unsup bundle) if user didn't pass --unsup_npz.
if args.unsup_npz is None:
    args.unsup_npz = os.path.join('data', 'unsupervised_data', 'test_data_smd_unsup.npz')

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from model.net import *
from trainer import Tester
from lib.logger import get_logger
from lib.dataloader_smd import load_data3, load_data_unsup_train
from lib.dataloader_swat import load_data as swat_load_data, load_data2 as swat_load_data2
from lib.dataloader_wadi import load_data as wadi_load_data, load_data2 as wadi_load_data2
from lib.utils import *
from model.utils import *

DEVICE = get_default_device()
args.device = DEVICE

paths = resolve_dataset_paths(
    args.data,
    data_root=args.data_root,
    dataset_root=args.dataset_root,
    group_name=args.group_name,
    train_file=args.train_file,
    test_file=args.test_file,
    unsup_npz=args.unsup_npz,
)


def _infer_nnodes_from_csv(train_csv_path: str) -> int:
    import pandas as pd
    df = pd.read_csv(train_csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    drop_cols = [c for c in ["Timestamp", "attack", "Attack", "Normal/Attack", "label", "Normal/Attack ", "Date", "Time", "Row", "Date_Time", "Date time"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    unnamed_like = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_like:
        df = df.drop(columns=unnamed_like)
    return df.shape[1]


def _extract_unsup_arrays(npz_obj, args):
    keys_local = set(npz_obj.keys())
    if {'a','b','c','d'}.issubset(keys_local):
        return npz_obj['a'], npz_obj['b'], npz_obj['c'], npz_obj['d']

    if {'a','b'}.issubset(keys_local) and 'c' not in keys_local and 'd' not in keys_local:
        attack = npz_obj['a']
        labels = npz_obj['b'].reshape(-1)
        n = len(labels)
        if args.unsup_train_size is not None:
            n_train = int(args.unsup_train_size)
        else:
            n_train = int(n * float(args.unsup_split))
        n_train = max(1, min(n_train, n - 1))
        return attack[:n_train], labels[:n_train], attack[n_train:], labels[n_train:]

    if {'attack_train','train_labels','attack_test','test_labels'}.issubset(keys_local):
        return npz_obj['attack_train'], npz_obj['train_labels'], npz_obj['attack_test'], npz_obj['test_labels']

    arr_keys = [k for k in keys_local if k.startswith('arr_')]
    if len(arr_keys) >= 4:
        arr_keys_sorted = sorted(arr_keys)
        return npz_obj[arr_keys_sorted[0]], npz_obj[arr_keys_sorted[1]], npz_obj[arr_keys_sorted[2]], npz_obj[arr_keys_sorted[3]]

    raise KeyError(
        "Unsupported npz format. Expected keys a/b/c/d or attack_train/train_labels/attack_test/test_labels (or a/b only). "
        f"Available keys: {sorted(keys_local)}"
    )


# ---------------- data loading (unsup) ----------------
data_lower = args.data.lower()

if data_lower in ["swat", "wadi"]:
    # Unsupervised mode for SWaT/WADI: train on normal csv, test on attack csv.
    if paths.train_csv is None or paths.test_csv is None:
        raise FileNotFoundError(f"train/test csv not resolved for {args.data}")

    inferred = _infer_nnodes_from_csv(paths.train_csv)
    if args.nnodes is None or args.nnodes != inferred:
        if args.nnodes is not None and args.nnodes != inferred:
            print(f"[WARN] args.nnodes={args.nnodes} but CSV has {inferred} feature columns. Using nnodes={inferred}.")
        args.nnodes = inferred

    if data_lower == "swat":
        if args.is_mas:
            train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = swat_load_data2(
                paths.train_csv, paths.test_csv,
                device=DEVICE,
                window_size=args.window_size,
                val_ratio=args.val_ratio,
                batch_size=args.batch_size,
                is_down_sample=args.is_down_sample,
                down_len=args.down_len,
            )
        else:
            train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = swat_load_data(
                paths.train_csv, paths.test_csv,
                device=DEVICE,
                window_size=args.window_size,
                val_ratio=args.val_ratio,
                batch_size=args.batch_size,
                is_down_sample=args.is_down_sample,
                down_len=args.down_len,
            )
    else:
        if args.is_mas:
            train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = wadi_load_data2(
                paths.train_csv, paths.test_csv,
                device=DEVICE,
                window_size=args.window_size,
                val_ratio=args.val_ratio,
                batch_size=args.batch_size,
                is_down_sample=args.is_down_sample,
                down_len=args.down_len,
            )
        else:
            train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = wadi_load_data(
                paths.train_csv, paths.test_csv,
                device=DEVICE,
                window_size=args.window_size,
                val_ratio=args.val_ratio,
                batch_size=args.batch_size,
                is_down_sample=args.is_down_sample,
                down_len=args.down_len,
            )

else:
    # NPZ unsupervised bundle (SMD/others)
    if paths.unsup_npz is None:
        raise ValueError("--unsup_npz is required for npz unsupervised testing")
    if not os.path.isfile(paths.unsup_npz):
        raise FileNotFoundError(f"Unsupervised npz not found: {paths.unsup_npz}")

    smd_unsup_data = np.load(paths.unsup_npz, allow_pickle=True)
    keys = set(smd_unsup_data.keys())
    print(f"[INFO] loaded unsup npz: {paths.unsup_npz}; keys={sorted(keys)}")

    attack_train, train_labels, attack_test, test_labels = _extract_unsup_arrays(smd_unsup_data, args)
    print(f"[INFO] unsup sizes: train={len(train_labels)}, test={len(test_labels)}")

    # test loader (attack_test + test_labels)
    _, _, test_loader, y_test_labels, _ = load_data3(
        attack_train,
        attack_test,
        test_labels,
        device=DEVICE,
        window_size=args.window_size,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        is_down_sample=args.is_down_sample,
        down_len=args.down_len,
    )

    # train loader for POT threshold (train errors)
    train_loader, val_loader, min_max_scaler = load_data_unsup_train(
        attack_train,
        train_labels,
        device=DEVICE,
        window_size=args.window_size,
        val_ratio=0.05,
        batch_size=args.batch_size,
        is_down_sample=args.is_down_sample,
        down_len=args.down_len,
    )


# ---------------- model + evaluation ----------------
init_seed(args.seed)

channels_list = [[16,8,32],[32,8,64]]
AE_IN_CHANNELS = args.window_size * args.nnodes * args.in_channels
latent_size = args.window_size * args.latent_size

pred_model = STATModel(args, DEVICE, args.window_size - args.n_pred, channels_list, static_feat=None)
ae_model = EncoderDecoder(AE_IN_CHANNELS, latent_size, AE_IN_CHANNELS, not args.real_value)

from lib.logger import get_logger, log_hparams
logger = get_logger(exp.log_dir, name=args.model, debug=args.debug, data=args.data, tag='test', model=args.model, run_id=args.run_id, console=True)
log_hparams(logger, args)

model_path = os.path.join(exp.pth_dir, 'best_model_' + args.data + "_" + args.model + '.pth')

tester = Tester(pred_model, ae_model, args, min_max_scaler, logger, path=model_path,
                alpha=args.test_alpha, beta=args.test_beta, gamma=args.test_gamma)

map_location = torch.device(DEVICE)


def _get_train_results(train_loader, map_location):
    train_results = tester.testing(train_loader, map_location)
    (train_y_pred,
     train_loss1_list,
     train_loss2_list,
     train_pred_list,
     train_gt_list,
     train_origin_list,
     train_construct_list,
     train_generate_list,
     train_generate_construct_list) = concate_results(train_results)

    train_pred_results = [train_pred_list, train_gt_list]
    train_ae_results = [train_construct_list, train_origin_list]
    train_generate_results = [train_generate_list, train_generate_construct_list]
    return train_pred_results, train_ae_results, train_generate_results


train_pred_results, train_ae_results, train_generate_results = _get_train_results(train_loader, map_location)

test_results = tester.testing(test_loader, map_location)

test_y_pred, test_loss1_list, test_loss2_list, test_pred_list, test_gt_list, test_origin_list, test_construct_list, test_generate_list, test_generate_construct_list = concate_results(test_results)

print("scores: ", len(test_y_pred), test_y_pred.mean())
print("loss1: ", len(test_loss1_list), test_loss1_list.mean())
print("loss2: ", len(test_loss2_list), test_loss2_list.mean())
print("y_pred: ", len(test_y_pred))
print("y_test_labels: ", len(y_test_labels))

test_pred_results = [test_pred_list, test_gt_list]
test_ae_results = [test_construct_list, test_origin_list]
test_generate_results = [test_generate_list, test_generate_construct_list]

print("================= Find best f1 from score: POT (method=max) =================")
info = get_final_result_POT(test_pred_results, test_ae_results, test_generate_results,
                            train_pred_results, train_ae_results, train_generate_results,
                            y_test_labels, topk=1, option=2, method="max",
                            alpha=args.test_alpha, beta=args.test_beta, gamma=args.test_gamma,
                            search_steps=args.search_steps)
print(info)

print("\n================= Find best f1 from score: POT (method=sum) =================")
info = get_final_result_POT(test_pred_results, test_ae_results, test_generate_results,
                            train_pred_results, train_ae_results, train_generate_results,
                            y_test_labels, topk=1, option=2, method="sum",
                            alpha=args.test_alpha, beta=args.test_beta, gamma=args.test_gamma,
                            search_steps=args.search_steps)
print(info)

print("\n================= Find best f1 from score: POT (method=mean) =================")
info = get_final_result_POT(test_pred_results, test_ae_results, test_generate_results,
                            train_pred_results, train_ae_results, train_generate_results,
                            y_test_labels, topk=1, option=2, method="mean",
                            alpha=args.test_alpha, beta=args.test_beta, gamma=args.test_gamma,
                            search_steps=args.search_steps)
print(info)
