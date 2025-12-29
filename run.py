import os
import argparse

from lib.cli import add_common_args, finalize_args
from lib.metrics import masked_mse_loss
from lib.paths import resolve_dataset_paths


parser = argparse.ArgumentParser(description='PyTorch Prediction Model on Time-series Dataset')
add_common_args(parser)
args = finalize_args(parser.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from model.net import *
from trainer import Trainer
from lib.dataloader_swat import load_data as swat_load_data, load_data2 as swat_load_data2
from lib.dataloader_wadi import load_data as wadi_load_data, load_data2 as wadi_load_data2
from lib.dataloader_smd import load_data as smd_load_data, load_data2 as smd_load_data2
from lib.dataloader_msl_smap import load_data as msl_load_data, load_data2 as msl_load_data2
from lib.utils import *
from model.utils import *


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


DEVICE = get_default_device()
args.device = DEVICE

paths = resolve_dataset_paths(
    args.data,
    data_root=args.data_root,
    dataset_root=args.dataset_root,
    group_name=args.group_name,
    train_file=args.train_file,
    test_file=args.test_file,
)

data_lower = args.data.lower()

# ---- Load data depending on dataset ----
if data_lower == "swat":
    if paths.train_csv is None or paths.test_csv is None:
        raise FileNotFoundError("SWaT train/test csv paths not resolved")
    inferred_nnodes = _infer_nnodes_from_csv(paths.train_csv)
    if args.nnodes is None or args.nnodes != inferred_nnodes:
        if args.nnodes is not None and args.nnodes != inferred_nnodes:
            print(f"[WARN] args.nnodes={args.nnodes} but SWaT CSV has {inferred_nnodes} feature columns. Using nnodes={inferred_nnodes}.")
        args.nnodes = inferred_nnodes

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

elif data_lower == "wadi":
    if paths.train_csv is None or paths.test_csv is None:
        raise FileNotFoundError("WADI train/test csv paths not resolved")
    inferred_nnodes = _infer_nnodes_from_csv(paths.train_csv)
    if args.nnodes is None or args.nnodes != inferred_nnodes:
        if args.nnodes is not None and args.nnodes != inferred_nnodes:
            print(f"[WARN] args.nnodes={args.nnodes} but WADI CSV has {inferred_nnodes} feature columns. Using nnodes={inferred_nnodes}.")
        args.nnodes = inferred_nnodes
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

elif data_lower == "smd":
    group_name = paths.group_name or "machine-1-1"
    if args.is_mas:
        train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = smd_load_data2(args.data, group_name,
                                                                                               device=DEVICE,
                                                                                               window_size=args.window_size,
                                                                                               val_ratio=args.val_ratio,
                                                                                               batch_size=args.batch_size,
                                                                                               is_down_sample=args.is_down_sample,
                                                                                               down_len=args.down_len)
    else:
        train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = smd_load_data(args.data, group_name,
                                                                                              device=DEVICE,
                                                                                              window_size=args.window_size,
                                                                                              val_ratio=args.val_ratio,
                                                                                              batch_size=args.batch_size,
                                                                                              is_down_sample=args.is_down_sample,
                                                                                              down_len=args.down_len)

elif data_lower in ["msl", "smap"]:
    if args.is_mas:
        train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = msl_load_data2(args.data,
                                                                                               device=DEVICE,
                                                                                               window_size=args.window_size,
                                                                                               val_ratio=args.val_ratio,
                                                                                               batch_size=args.batch_size,
                                                                                               is_down_sample=args.is_down_sample,
                                                                                               down_len=args.down_len)
    else:
        train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = msl_load_data(args.data,
                                                                                              device=DEVICE,
                                                                                              window_size=args.window_size,
                                                                                              val_ratio=args.val_ratio,
                                                                                              batch_size=args.batch_size,
                                                                                              is_down_sample=args.is_down_sample,
                                                                                              down_len=args.down_len)
else:
    raise ValueError(f"Unsupported dataset: {args.data}")

# ---- Training ----
init_seed(args.seed)

channels_list = [[16,8,32],[32,8,64]]
AE_IN_CHANNELS = args.window_size * args.nnodes * args.in_channels
latent_size = args.window_size * args.latent_size

pred_model = STATModel(args, DEVICE, args.window_size - args.n_pred, channels_list, static_feat=None)
pred_model = to_device(pred_model, DEVICE)
pred_optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=args.pred_lr_init, eps=1.0e-8, weight_decay=0.0001, amsgrad=False)
pred_loss = masked_mse_loss(mask_value=-0.01)

ae_model = EncoderDecoder(AE_IN_CHANNELS, latent_size, AE_IN_CHANNELS, not args.real_value)
ae_model = to_device(ae_model, DEVICE)
ae_optimizer = torch.optim.Adam(params=ae_model.parameters(), lr=args.ae_lr_init, eps=1.0e-8, weight_decay=0.0001, amsgrad=False)
ae_loss = masked_mse_loss(mask_value=-0.01)

print_model_parameters(pred_model)
print_model_parameters(ae_model)

trainer = Trainer(pred_model, pred_loss, pred_optimizer, ae_model, ae_loss, ae_optimizer,
                  train_loader, val_loader, test_loader, args, min_max_scaler, lr_scheduler=None)

train_history, val_history = trainer.train()

plot_history(train_history, model=args.model, mode="train", data=args.data)
plot_history(val_history, model=args.model, mode="val", data=args.data)
plot_history2(val_history, model=args.model, mode="val", data=args.data)
