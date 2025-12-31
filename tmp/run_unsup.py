import os
import argparse

from lib.cli import add_common_args, finalize_args
from lib.metrics import masked_mse_loss
from lib.paths import resolve_dataset_paths


parser = argparse.ArgumentParser(description='PyTorch Prediction Model on Time-series Dataset')
add_common_args(parser)

args = finalize_args(parser.parse_args())

# Provide a reasonable default for NPZ mode (SMD unsup bundle) if user didn't pass --unsup_npz.
if args.unsup_npz is None:
    args.unsup_npz = os.path.join('data', 'unsupervised_data', 'test_data_smd_unsup.npz')

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from model.net import *
from trainer import Trainer
from lib.dataloader_smd import load_data3, load_data_unsup_train
from lib.dataloader_swat import load_data as swat_load_data, load_data2 as swat_load_data2
from lib.dataloader_wadi import load_data as wadi_load_data, load_data2 as wadi_load_data2
from lib.utils import *
from model.utils import *

DEVICE = get_default_device()
args.device = DEVICE

paths = resolve_dataset_paths(args.data,
                             data_root=args.data_root,
                             dataset_root=args.dataset_root,
                             group_name=args.group_name,
                             train_file=args.train_file,
                             test_file=args.test_file,
                             unsup_npz=args.unsup_npz)


def _infer_nnodes_from_csv(train_csv_path: str) -> int:
    import pandas as pd
    df = pd.read_csv(train_csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    drop_cols = [c for c in ["Timestamp", "attack", "Attack", "Normal/Attack", "label", "Normal/Attack "] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    unnamed_like = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_like:
        df = df.drop(columns=unnamed_like)
    return df.shape[1]

data_lower = args.data.lower()

if data_lower in ["swat", "wadi"]:
    # "Unsupervised" training for SWaT/WADI in this repo typically means
    # training on normal (train csv) and evaluating on attack (test csv).
    if paths.train_csv is None or paths.test_csv is None:
        raise FileNotFoundError(f"train/test csv not resolved for {args.data}")
    if data_lower == "swat":
        inferred = _infer_nnodes_from_csv(paths.train_csv)
        if args.nnodes != inferred:
            print(f"[WARN] args.nnodes={args.nnodes} but SWaT CSV has {inferred} feature columns. Using nnodes={inferred}.")
            args.nnodes = inferred

    if data_lower == "swat":
        if args.is_mas:
            train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = swat_load_data2(paths.train_csv, paths.test_csv,
                                                                                                   device=DEVICE,
                                                                                                   window_size=args.window_size,
                                                                                                   val_ratio=args.val_ratio,
                                                                                                   batch_size=args.batch_size,
                                                                                                   is_down_sample=args.is_down_sample,
                                                                                                   down_len=args.down_len)
        else:
            train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = swat_load_data(paths.train_csv, paths.test_csv,
                                                                                                  device=DEVICE,
                                                                                                  window_size=args.window_size,
                                                                                                  val_ratio=args.val_ratio,
                                                                                                  batch_size=args.batch_size,
                                                                                                  is_down_sample=args.is_down_sample,
                                                                                                  down_len=args.down_len)
    else:
        if args.is_mas:
            train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = wadi_load_data2(paths.train_csv, paths.test_csv,
                                                                                                   device=DEVICE,
                                                                                                   window_size=args.window_size,
                                                                                                   val_ratio=args.val_ratio,
                                                                                                   batch_size=args.batch_size,
                                                                                                   is_down_sample=args.is_down_sample,
                                                                                                   down_len=args.down_len)
        else:
            train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = wadi_load_data(paths.train_csv, paths.test_csv,
                                                                                                  device=DEVICE,
                                                                                                  window_size=args.window_size,
                                                                                                  val_ratio=args.val_ratio,
                                                                                                  batch_size=args.batch_size,
                                                                                                  is_down_sample=args.is_down_sample,
                                                                                                  down_len=args.down_len)

else:
    if paths.unsup_npz is None:
        raise ValueError("--unsup_npz is required for npz unsupervised training")
    if not os.path.isfile(paths.unsup_npz):
        raise FileNotFoundError(f"Unsupervised npz not found: {paths.unsup_npz}. Set --unsup_npz to a valid path.")

    smd_unsup_data = np.load(paths.unsup_npz, allow_pickle=True)
    keys = set(smd_unsup_data.keys())
    print(f"[INFO] loaded unsup npz: {paths.unsup_npz}; keys={sorted(keys)}")

    def _extract_unsup_arrays(npz_obj):
        keys_local = set(npz_obj.keys())
        # Common pattern a/b/c/d
        if {'a','b','c','d'}.issubset(keys_local):
            return npz_obj['a'], npz_obj['b'], npz_obj['c'], npz_obj['d']
        # Minimal pattern a/b only: attack data + labels. We'll split into train/test.
        if {'a','b'}.issubset(keys_local) and 'c' not in keys_local and 'd' not in keys_local:
            attack = npz_obj['a']
            labels = npz_obj['b']
            # ensure 1d labels
            labels = labels.reshape(-1)
            n = len(labels)
            if args.unsup_train_size is not None:
                n_train = int(args.unsup_train_size)
            else:
                n_train = int(n * float(args.unsup_split))
            n_train = max(1, min(n_train, n - 1))
            return attack[:n_train], labels[:n_train], attack[n_train:], labels[n_train:]
        # Named arrays
        name_map = {'attack_train':'attack_train', 'train_labels':'train_labels', 'attack_test':'attack_test', 'test_labels':'test_labels'}
        if set(name_map.keys()).issubset(keys_local):
            return npz_obj['attack_train'], npz_obj['train_labels'], npz_obj['attack_test'], npz_obj['test_labels']
        # Generic arr_0..arr_3
        arr_keys = [k for k in keys_local if k.startswith('arr_')]
        if len(arr_keys) >= 4:
            arr_keys_sorted = sorted(arr_keys)
            return npz_obj[arr_keys_sorted[0]], npz_obj[arr_keys_sorted[1]], npz_obj[arr_keys_sorted[2]], npz_obj[arr_keys_sorted[3]]
        raise KeyError(
            "Unsupported npz format. Expected keys a/b/c/d or attack_train/train_labels/attack_test/test_labels (or a/b only). "
            f"Available keys: {sorted(keys_local)}"
        )

    attack_train, train_labels, attack_test, test_labels = _extract_unsup_arrays(smd_unsup_data)
    print(len(test_labels))

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

## set seed
init_seed(args.seed)

channels_list = [[16,8,32],[32,8,64]]

AE_IN_CHANNELS = args.window_size * args.nnodes * args.in_channels
latent_size = args.window_size * args.latent_size


pred_model = STATModel(args, DEVICE, args.window_size - args.n_pred, channels_list, static_feat=None)

pred_model = to_device(pred_model, DEVICE)
pred_optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=args.pred_lr_init, eps=1.0e-8, weight_decay=args.pred_weight_decay, amsgrad=False)
pred_loss = masked_mse_loss(mask_value = -0.01)


ae_model = EncoderDecoder(AE_IN_CHANNELS, latent_size, AE_IN_CHANNELS, not args.real_value)
ae_model = to_device(ae_model, DEVICE)
ae_optimizer = torch.optim.Adam(params=ae_model.parameters(), lr=args.ae_lr_init, eps=1.0e-8, weight_decay=args.ae_weight_decay, amsgrad=False)
ae_loss = masked_mse_loss(mask_value = -0.01)


trainer = Trainer(pred_model, pred_loss, pred_optimizer, ae_model, ae_loss, ae_optimizer, train_loader, val_loader, test_loader, args, min_max_scaler, lr_scheduler=None)

train_history, val_history = trainer.train()

plot_history(train_history, model = args.model, mode="train", data=args.data)
plot_history(val_history, model = args.model, mode="val", data=args.data)
plot_history2(val_history, model = args.model, mode="val", data=args.data)
