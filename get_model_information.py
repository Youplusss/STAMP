import os
import argparse

from lib.cli import add_common_args, finalize_args
from lib.paths import resolve_dataset_paths


parser = argparse.ArgumentParser(description='PyTorch Prediction Model on Time-series Dataset')
add_common_args(parser)
args = finalize_args(parser.parse_args())
# get_model_information is NPZ-based; keep defaults close to previous behavior
if args.unsup_train_size is None:
    args.unsup_train_size = 15000

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from model.net import *
from trainer import Trainer, Tester
from lib.logger import get_logger
from lib.dataloader_smd import load_data3
from lib.utils import *
from lib.metrics import *
from model.utils import *
from lib.evaluate import *

DEVICE = get_default_device()

paths = resolve_dataset_paths(args.data, group_name=args.group_name, train_file=args.train_file, test_file=args.test_file, unsup_npz=args.unsup_npz)


def _load_unsup_npz(npz_path: str):
    if npz_path is None:
        raise ValueError("--unsup_npz is required")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"unsup npz not found: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)
    keys = set(npz.keys())
    print(f"[INFO] loaded unsup npz: {npz_path}; keys={sorted(keys)}")
    if {'a','b','c','d'}.issubset(keys):
        return npz['a'], npz['b'], npz['c'], npz['d']
    if {'a','b'}.issubset(keys) and 'c' not in keys and 'd' not in keys:
        attack = npz['a']
        labels = npz['b'].reshape(-1)
        n_train = int(args.unsup_train_size)
        n_train = max(1, min(n_train, len(labels) - 1))
        return attack[:n_train], labels[:n_train], attack[n_train:], labels[n_train:]
    arr_keys = [k for k in keys if k.startswith('arr_')]
    if len(arr_keys) >= 4:
        arr = sorted(arr_keys)
        return npz[arr[0]], npz[arr[1]], npz[arr[2]], npz[arr[3]]
    raise KeyError(f"Unsupported npz format; keys={sorted(keys)}")


attack_train, train_labels, attack_test, test_labels = _load_unsup_npz(paths.unsup_npz)

train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = load_data3(attack_train, attack_test, test_labels,
                                                                                    device=DEVICE,
                                                                                    window_size=args.window_size,
                                                                                    val_ratio=0.05,
                                                                                    batch_size=args.batch_size,
                                                                                    is_down_sample=args.is_down_sample,
                                                                                    down_len=args.down_len)




## set seed
init_seed(args.seed)

channels_list = [[16,8,32],[32,8,64]]

AE_IN_CHANNELS = args.window_size * args.nnodes * args.in_channels
latent_size = args.window_size * args.latent_size


pred_model = STATModel(args, DEVICE, args.window_size - args.n_pred, channels_list, static_feat=None)


pred_model = to_device(pred_model, DEVICE)
pred_optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=args.pred_lr_init, eps=1.0e-8, weight_decay=0.0001, amsgrad=False)
pred_loss = masked_mse_loss(mask_value = -0.01)


ae_model = EncoderDecoder(AE_IN_CHANNELS, latent_size, AE_IN_CHANNELS, not args.real_value)
ae_model = to_device(ae_model, DEVICE)
ae_optimizer = torch.optim.Adam(params=ae_model.parameters(), lr=args.ae_lr_init, eps=1.0e-8, weight_decay=0.0001, amsgrad=False)
ae_loss = masked_mse_loss(mask_value = -0.01)




trainer = Trainer(pred_model, pred_loss, pred_optimizer, ae_model, ae_loss, ae_optimizer, train_loader, val_loader, test_loader, args, min_max_scaler, lr_scheduler=None)

train_history, val_history = trainer.train()


from lib.paths import resolve_experiment_dirs

# resolve experiment dirs early
exp = resolve_experiment_dirs(getattr(args, 'log_dir', 'expe'))
args.run_id = exp.run_id
args.log_dir = exp.root
args.log_dir_log = exp.log_dir
args.log_dir_pth = exp.pth_dir
args.log_dir_pdf = exp.pdf_dir

# logger
from lib.logger import get_logger, log_hparams
logger = get_logger(exp.log_dir, name=args.model, debug=args.debug, data=args.data, tag='info', model=args.model, run_id=args.run_id, console=True)
log_hparams(logger, args)

# checkpoint path
model_path = os.path.join(exp.pth_dir, 'best_model_' + args.data + "_" + args.model + '.pth')


tester = Tester(pred_model, ae_model, args, min_max_scaler, logger, path=model_path, alpha=args.test_alpha, beta=args.test_beta, gamma=args.test_gamma)

map_location = torch.device(DEVICE)
# map_location = lambda storage.cuda(0), loc: storage
##[val_gt_list, val_pred_list, val_construct_list]
    

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



# get model information (three types of feature importance)
check_point = torch.load(model_path, map_location=map_location, weights_only=False)
pred_state_dict = check_point['pred_state_dict']

pred_model.load_state_dict(pred_state_dict)
print("load pred model done!")
pred_model.to(DEVICE)

target_num = args.nnodes
sort_graph_weight_out, sort_graph_weight_in = get_graph_weight(pred_model, args.nnodes, target_num)
sort_score_weight = get_score_weight(test_pred_results, test_ae_results,  test_generate_results, y_test_labels, topk = 1, option = 2, method="max", alpha =args.test_alpha, beta=args.test_beta, gamma = args.test_gamma, target_num=target_num)
os.makedirs('weights', exist_ok=True)
np.savez(os.path.join('weights', f'node_weights_{args.data}_unsup_train_STAMP.npz'), a=sort_graph_weight_out, b=sort_graph_weight_in, c=sort_score_weight)
