# -*- coding: utf-8 -*-

import os
import argparse

import torch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='STAMP Training (支持替换为 Mamba 预测/重构分支)')

    # -------------------------- basic --------------------------
    parser.add_argument('--data', type=str, default='SWaT',
                        help='dataset name (SWaT, WADI, SMD, SMAP, MSL, ...)')
    parser.add_argument('--debug', default=False, type=eval)
    parser.add_argument('--real_value', default=False, type=eval)
    parser.add_argument('--log_dir', default="expe", type=str)
    # 兼容 Trainer.transfer_path（老代码会用到 log_dir_transfer）
    parser.add_argument('--log_dir_transfer', default=None, type=str,
                        help='(optional) transfer checkpoint dir; default=log_dir')

    parser.add_argument('--model', default="v2_", type=str)
    # pred_model: gat | mamba
    parser.add_argument('--pred_model', default="mamba", type=str,
                        help="prediction branch type: gat | mamba")
    # recon_model: ae | mamba
    parser.add_argument('--recon_model', default="mamba", type=str,
                        help="reconstruction branch type: ae | mamba")

    parser.add_argument('--gpu_id', default="0", type=str)
    parser.add_argument('--temp_method', default="SAttn", type=str)

    # SMD 需要 group_name
    parser.add_argument('--group', type=str, default='machine-1-1',
                        help='SMD group name, e.g. machine-1-1')

    # SWaT/WADI 需要文件名（如果你用自定义路径，可直接传绝对路径）
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)

    # -------------------------- graph / STAMP-GAT params --------------------------
    parser.add_argument('--nnodes', type=int, default=45, help='number of nodes/features')
    parser.add_argument('--top_k', type=int, default=10, help='top-k neighbors for graph')
    parser.add_argument('--em_dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--alpha', type=int, default=3, help='alpha')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden_dim')
    parser.add_argument('--att_option', type=int, default=1, help='att_option')

    # -------------------------- sequence params --------------------------
    parser.add_argument('--window_size', type=int, default=15, help='window_size')
    parser.add_argument('--n_pred', type=int, default=3, help='n_pred (forecast horizon)')
    parser.add_argument('--temp_kernel', type=int, default=5, help='temp_kernel')
    parser.add_argument('--in_channels', type=int, default=1, help='in_channels (raw channels)')
    parser.add_argument('--out_channels', type=int, default=1, help='out_channels')

    parser.add_argument('--layer_num', type=int, default=2, help='layer_num (for STAMP-GAT)')
    parser.add_argument('--act_func', type=str, default="GLU", help='act_func')
    parser.add_argument('--pred_lr_init', type=float, default=0.001, help='pred_lr_init')

    # -------------------------- attention params (STAMP-GAT) --------------------------
    parser.add_argument('--embed_size', type=int, default=64, help='embed_size')
    parser.add_argument('--num_heads', type=int, default=8, help='num_heads')
    parser.add_argument('--num_layers', type=int, default=1, help='num_attn_layers')
    parser.add_argument('--ffwd_size', type=int, default=32, help='ffwd_size')
    parser.add_argument('--is_conv', type=eval, default=False)
    parser.add_argument('--return_weight', type=eval, default=False)

    # -------------------------- AE (old) params --------------------------
    parser.add_argument('--latent_size', type=int, default=1, help='latent_size (old AE)')
    parser.add_argument('--ae_lr_init', type=float, default=0.001, help='ae_lr_init')

    # -------------------------- training params --------------------------
    parser.add_argument('--seed', type=int, default=666, help='seed')
    parser.add_argument('--val_ratio', type=float, default=.2, help='val_ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epoch')

    # Whether to use the coupled/adversarial objectives (loss1/loss2) in addition to plain pred_loss/ae_loss.
    # For some models (notably mamba recon), the adversarial terms can destabilize training after a few epochs.
    parser.add_argument('--use_adv', default=True, type=eval,
                        help='use coupled/adversarial training objectives (default True; set False for stable mamba-mamba)')

    parser.add_argument('--is_down_sample', type=eval, default=True, help='down-sample raw series or not')
    parser.add_argument('--down_len', type=int, default=100, help='down sample ratio')

    # 对抗/耦合训练中是否冻结另一分支参数（推荐 True，显存/速度更友好）
    parser.add_argument('--adv_freeze_other', type=eval, default=True)

    # -------------------------- lr decay / early stop (kept for compatibility) --------------------------
    parser.add_argument('--early_stop', default=True, type=eval)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--grad_clip', default=True, type=eval, help='whether to clip gradients (recommended for Mamba)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm for clip_grad_norm_')

    parser.add_argument('--lr_decay', default=True, type=eval)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_step', default="5,20,40,70", type=str)
    parser.add_argument('--largest_loss_diff', default=0.2, type=float)

    # -------------------------- moving average (MAS) --------------------------
    parser.add_argument('--is_mas', default=True, type=eval,
                        help='perform moving average to extend channels (MAS)')

    # -------------------------- Mamba Forecast params --------------------------
    parser.add_argument('--mamba_use_mas', default=True, type=eval,
                        help='(only for pred_model=mamba) whether to use MAS channels as extra tokens')
    parser.add_argument('--mamba_d_model', type=int, default=256)
    parser.add_argument('--mamba_e_layers', type=int, default=3)
    parser.add_argument('--mamba_d_state', type=int, default=16)
    parser.add_argument('--mamba_d_conv', type=int, default=4)
    parser.add_argument('--mamba_expand', type=int, default=2)
    parser.add_argument('--mamba_dropout', type=float, default=0.1)
    parser.add_argument('--mamba_use_norm', default=True, type=eval)
    parser.add_argument('--mamba_use_last_residual', default=True, type=eval)

    # -------------------------- Mamba Recon params --------------------------
    parser.add_argument('--recon_d_model', type=int, default=256)
    parser.add_argument('--recon_num_layers', type=str, default='2,2,2',
                        help="e.g. '2,2,2' for [local, mid, global] layers")
    parser.add_argument('--recon_d_state', type=int, default=16)
    parser.add_argument('--recon_d_conv', type=int, default=4)
    parser.add_argument('--recon_expand', type=int, default=2)
    parser.add_argument('--recon_dropout', type=float, default=0.1)
    parser.add_argument('--recon_output_activation', type=str, default='auto',
                        help='auto | none | sigmoid | tanh. auto: real_value=False -> sigmoid (recommended for MinMax [0,1])')

    return parser


def load_dataloaders(args, device):
    """根据 args.data 选择对应的 dataloader。"""

    base_dir = os.getcwd()
    data_name = args.data.upper()

    if data_name in ["SWAT", "WADI"]:
        if data_name == "SWAT":
            from lib.dataloader_swat import load_data, load_data2
            default_train = os.path.join(base_dir, "dataset", args.data, "swat_train.csv")
            default_test = os.path.join(base_dir, "dataset", args.data, "swat_test.csv")
        else:
            from lib.dataloader_wadi import load_data, load_data2
            default_train = os.path.join(base_dir, "dataset", args.data, "wadi_train.csv")
            default_test = os.path.join(base_dir, "dataset", args.data, "wadi_test.csv")

        train_file = args.train_file or default_train
        test_file = args.test_file or default_test

        if args.is_mas:
            return load_data2(
                train_file,
                test_file,
                device=device,
                window_size=args.window_size,
                val_ratio=args.val_ratio,
                batch_size=args.batch_size,
                is_down_sample=args.is_down_sample,
                down_len=args.down_len,
            )
        return load_data(
            train_file,
            test_file,
            device=device,
            window_size=args.window_size,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            is_down_sample=args.is_down_sample,
            down_len=args.down_len,
        )

    if data_name == "SMD":
        # New CSV-mode SMD (generated by scripts/process_smd.py)
        from lib.paths import resolve_dataset_paths
        from lib.dataloader_smd import load_data_csv, load_data2_csv

        paths = resolve_dataset_paths(
            args.data,
            base_dir=base_dir,
            data_root=None,
            dataset_root=None,
            group_name=getattr(args, 'group', None),
            train_file=getattr(args, 'train_file', None),
            test_file=getattr(args, 'test_file', None),
        )

        if paths.train_csv is None or paths.test_csv is None:
            raise FileNotFoundError("SMD CSV paths not resolved; run scripts/process_smd.py first")

        if args.is_mas:
            return load_data2_csv(
                paths.train_csv,
                paths.test_csv,
                device=device,
                window_size=args.window_size,
                val_ratio=args.val_ratio,
                batch_size=args.batch_size,
                is_down_sample=args.is_down_sample,
                down_len=args.down_len,
            )
        return load_data_csv(
            paths.train_csv,
            paths.test_csv,
            device=device,
            window_size=args.window_size,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            is_down_sample=args.is_down_sample,
            down_len=args.down_len,
        )

    if data_name in ["MSL", "SMAP"]:
        from lib.paths import resolve_dataset_paths
        from lib.dataloader_msl_smap import load_data, load_data2, load_data_csv, load_data2_csv

        paths = resolve_dataset_paths(
            args.data,
            base_dir=base_dir,
            data_root=None,
            dataset_root=None,
            train_file=getattr(args, 'train_file', None),
            test_file=getattr(args, 'test_file', None),
        )

        # Prefer CSV mode when train/test CSV exist
        if paths.train_csv and paths.test_csv and os.path.isfile(paths.train_csv) and os.path.isfile(paths.test_csv):
            if args.is_mas:
                return load_data2_csv(
                    paths.train_csv,
                    paths.test_csv,
                    device=device,
                    window_size=args.window_size,
                    val_ratio=args.val_ratio,
                    batch_size=args.batch_size,
                    is_down_sample=args.is_down_sample,
                    down_len=args.down_len,
                )
            return load_data_csv(
                paths.train_csv,
                paths.test_csv,
                device=device,
                window_size=args.window_size,
                val_ratio=args.val_ratio,
                batch_size=args.batch_size,
                is_down_sample=args.is_down_sample,
                down_len=args.down_len,
            )

        # Legacy PKL mode fallback
        if args.is_mas:
            return load_data2(
                data_name,
                device=device,
                window_size=args.window_size,
                val_ratio=args.val_ratio,
                batch_size=args.batch_size,
                is_down_sample=args.is_down_sample,
                down_len=args.down_len,
            )
        return load_data(
            data_name,
            device=device,
            window_size=args.window_size,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            is_down_sample=args.is_down_sample,
            down_len=args.down_len,
        )

    raise ValueError(f"Unsupported dataset: {args.data}")


def infer_and_override_data_shape(args, train_loader):
    """尽可能从数据中推断 nnodes/in_channels，避免 shape mismatch。"""
    try:
        x = train_loader.dataset.x
        # x: [num_windows, T, N, C]
        N = int(x.shape[2])
        C = int(x.shape[3])
        if getattr(args, 'nnodes', None) != N:
            print(f"[Info] Override args.nnodes: {args.nnodes} -> {N} (from data)")
            args.nnodes = N
        if getattr(args, 'in_channels', None) != C:
            print(f"[Info] Override args.in_channels: {args.in_channels} -> {C} (from data)")
            args.in_channels = C
        # out_channels 默认与 in_channels 一致（STAMP 默认 1）
        if getattr(args, 'out_channels', None) != C:
            # 这里不强制覆盖 out_channels，除非用户没显式改过
            pass
    except Exception as e:
        print(f"[Warn] Failed to infer nnodes/in_channels from dataset: {e}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Name: v2_mamba / v2_gat...
    args.model = args.model + args.pred_model

    # Resolve experiment directories
    from lib.paths import resolve_experiment_dirs
    exp = resolve_experiment_dirs(args.log_dir)
    args.run_id = exp.run_id
    args.log_dir = exp.root
    args.log_dir_transfer = exp.root
    args.log_dir_log = exp.log_dir
    args.log_dir_pth = exp.pth_dir
    args.log_dir_pdf = exp.pdf_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    from lib.utils import get_default_device, to_device, plot_history, plot_history2
    from lib.metrics import masked_mse_loss
    from lib.logger import get_logger, log_hparams
    from model.utils import print_model_parameters, init_seed

    # device
    print(torch.cuda.is_available())
    device = get_default_device()
    args.device = device

    # load data
    train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = load_dataloaders(args, device)

    # infer shape
    infer_and_override_data_shape(args, train_loader)

    # seed
    init_seed(args.seed)

    # build models
    pred_model_type = args.pred_model.lower()
    recon_model_type = getattr(args, 'recon_model', 'ae').lower()

    if pred_model_type == 'mamba' or recon_model_type == 'mamba':
        from model.mamba_wrappers import build_stamp_mamba_models
        mamba_pred, mamba_ae = build_stamp_mamba_models(args)
    else:
        mamba_pred, mamba_ae = None, None

    # pred branch
    if pred_model_type == 'mamba':
        pred_model = mamba_pred
    else:
        from model.net import STATModel
        channels_list = [[16, 8, 32], [32, 8, 64]]
        pred_model = STATModel(args, device, args.window_size - args.n_pred, channels_list, static_feat=None)

    # recon branch
    if recon_model_type == 'mamba':
        ae_model = mamba_ae
    else:
        from model.net import EncoderDecoder
        AE_IN_CHANNELS = args.window_size * args.nnodes * args.in_channels
        latent_size = args.window_size * args.latent_size
        ae_model = EncoderDecoder(AE_IN_CHANNELS, latent_size, AE_IN_CHANNELS, not args.real_value)

    # move to device
    pred_model = to_device(pred_model, device)
    ae_model = to_device(ae_model, device)

    # optimizers / losses
    pred_optimizer = torch.optim.Adam(pred_model.parameters(), lr=args.pred_lr_init, eps=1.0e-8, weight_decay=1e-4)
    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=args.ae_lr_init, eps=1.0e-8, weight_decay=1e-4)

    pred_loss = masked_mse_loss(mask_value=-0.01)
    ae_loss = masked_mse_loss(mask_value=-0.01)

    # logger (file only; keep tqdm progress on console)
    logger = get_logger(
        exp.log_dir,
        name=args.model,
        debug=args.debug,
        data=args.data,
        tag='train',
        model=args.model,
        run_id=exp.run_id,
        console=True,
    )
    log_hparams(logger, args)

    # print params (parameter dump is kept as requested)
    print_model_parameters(pred_model)
    print_model_parameters(ae_model)

    # Trainer
    from trainer import Trainer
    trainer = Trainer(
        pred_model,
        pred_loss,
        pred_optimizer,
        ae_model,
        ae_loss,
        ae_optimizer,
        train_loader,
        val_loader,
        test_loader,
        args,
        min_max_scaler,
        lr_scheduler=None,
    )

    train_history, val_history = trainer.train()

    # plots -> expe/pdf
    plot_history(train_history, model=args.model, mode="train", data=args.data, out_dir=exp.pdf_dir, show=False)
    plot_history(val_history, model=args.model, mode="val", data=args.data, out_dir=exp.pdf_dir, show=False)
    plot_history2(val_history, model=args.model, mode="val", data=args.data, out_dir=exp.pdf_dir, show=False)


if __name__ == '__main__':
    main()
