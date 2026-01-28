import argparse

# CLI_VERSION 用于快速判断远端代码是否更新到包含 recon_model 的版本
CLI_VERSION = "2025-12-29-recon-model"


def _parse_use_adv(v):
    """Parse --use_adv flag.

    Supports:
    - 0/1/2 (recommended)
    - True/False (backward compatible): True -> 2, False -> 0
    """
    if isinstance(v, bool):
        return 2 if v else 0
    if isinstance(v, (int, float)):
        iv = int(v)
        if iv in (0, 1, 2):
            return iv
        raise argparse.ArgumentTypeError("--use_adv must be 0/1/2 or True/False")
    s = str(v).strip().lower()
    if s in ("0", "false", "no", "off"):
        return 0
    if s in ("1",):
        return 1
    if s in ("2", "true", "yes", "on"):
        return 2
    raise argparse.ArgumentTypeError("--use_adv must be 0/1/2 or True/False")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments shared by train/test and sup/unsup entrypoints."""

    # Dataset + paths
    parser.add_argument('--data', type=str, default='SWaT', help='dataset name (SWaT, WADI, SMD, MSL, SMAP, ...)')
    parser.add_argument('--data_root', type=str, default=None, help='override repo ./data directory')
    parser.add_argument('--dataset_root', type=str, default=None, help='override repo ./dataset directory (csv datasets)')
    parser.add_argument('--train_file', type=str, default=None, help='override train file path (csv datasets)')
    parser.add_argument('--test_file', type=str, default=None, help='override test file path (csv datasets)')
    parser.add_argument('--group_name', type=str, default=None, help='SMD subset name, e.g. machine-1-1')
    # tmp/ 版本里常用 --group，这里做兼容别名（finalize_args 会自动映射）
    parser.add_argument('--group', type=str, default=None, help='(alias) same as --group_name')

    # Runtime
    parser.add_argument('--debug', default=False, type=eval)
    parser.add_argument('--real_value', default=False, type=eval)
    parser.add_argument('--log_dir', default="expe", type=str)
    # 兼容 Trainer.transfer_path（可选）
    parser.add_argument('--log_dir_transfer', default=None, type=str,
                        help='(optional) transfer checkpoint dir; default=log_dir')
    parser.add_argument('--gpu_id', default="0", type=str)

    # Model choice
    parser.add_argument('--model', default="v2_", type=str)
    # pred_model: gat | mamba
    parser.add_argument('--pred_model', default="gat", type=str)
    # recon_model: ae | mamba
    parser.add_argument('--recon_model', default="ae", type=str,
                        help='reconstruction branch type: ae | mamba')
    parser.add_argument('--temp_method', default="SAttn", type=str)

    # Graph
    parser.add_argument('--nnodes', type=int, default=None, help='number of variables/features; inferred for SWaT when possible')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--em_dim', type=int, default=32)
    parser.add_argument('--alpha', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--att_option', type=int, default=1)

    # Pred model
    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--n_pred', type=int, default=3)
    parser.add_argument('--temp_kernel', type=int, default=5)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=1)

    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default="GLU")
    parser.add_argument('--pred_lr_init', type=float, default=0.001)

    # Attention
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--ffwd_size', type=int, default=32)
    parser.add_argument('--is_conv', type=eval, default=False)
    parser.add_argument('--return_weight', type=eval, default=False)

    # AE
    parser.add_argument('--latent_size', type=int, default=1)
    parser.add_argument('--ae_lr_init', type=float, default=0.001)

    # Training
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--val_ratio', type=float, default=.2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    # Back-compat alias for old README commands
    parser.add_argument('--epoch', type=int, default=None, help='alias for --epochs')

    # Testing weights
    parser.add_argument('--test_alpha', type=float, default=.5)
    # beta: reconstruction error weight. If you set beta=0, recon branch won't affect final anomaly score.
    parser.add_argument('--test_beta', type=float, default=.5)
    parser.add_argument('--test_gamma', type=float, default=0.5)
    parser.add_argument('--search_steps', default=50, type=int)

    # Unsupervised data (npz)
    parser.add_argument('--unsup_npz', type=str, default=None, help='path to unsupervised npz (a/b or a/b/c/d)')
    parser.add_argument('--unsup_split', type=float, default=0.7, help='train split ratio if npz only has a/b')
    parser.add_argument('--unsup_train_size', type=int, default=None, help='optional fixed train size if npz only has a/b')

    parser.add_argument('--is_down_sample', type=eval, default=True)
    parser.add_argument('--down_len', type=int, default=100)

    # Early stop / LR schedule
    parser.add_argument('--early_stop', default=True, type=eval)
    parser.add_argument('--early_stop_patience', type=int, default=10)

    parser.add_argument('--lr_decay', default=True, type=eval)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_step', default="5,20,40,70", type=str)

    # Misc
    parser.add_argument('--largest_loss_diff', default=0.2, type=float)
    parser.add_argument('--is_graph', default=True, type=eval)
    parser.add_argument('--is_mas', default=True, type=eval)

    # Coupled/adv training behavior
    parser.add_argument('--adv_freeze_other', type=eval, default=True,
                        help='freeze the other branch during adversarial/coupled update to save memory')

    # ---- Stabilized adversarial training (recommended for Mamba models) ----
    parser.add_argument('--use_adv', type=_parse_use_adv, default=2,
                        help='training loss mode: 0=no adversarial (weighted sum baseline), 1=legacy4 (tmp/trainer.py original), 2=current stabilized design. '
                             'Also accepts True/False for backward compatibility (True->2, False->0).')

    # baseline (use_adv=0) weights
    parser.add_argument('--loss_weight_pred', type=float, default=1.0,
                        help='(use_adv=0) weight for prediction loss')
    parser.add_argument('--loss_weight_ae', type=float, default=1.0,
                        help='(use_adv=0) weight for reconstruction loss')

    # AE / recon loss weights (used when not using adversarial training)
    parser.add_argument('--pred_loss_weight', type=float, default=1.0,
                        help='(use_adv=0) weight for prediction loss (legacy)')
    parser.add_argument('--ae_loss_weight', type=float, default=1.0,
                        help='(use_adv=0) weight for reconstruction loss (legacy)')

    # ---- Shared between pred and recon models ----
    parser.add_argument('--shared_emb', default=False, type=eval,
                        help='(legacy) use shared embedding between pred and recon models')
    parser.add_argument('--shared_attn', default=False, type=eval,
                        help='(legacy) use shared attention between pred and recon models')

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
    parser.add_argument('--recon_output_activation', type=str, default='none',
                        help='none | sigmoid | tanh')

    return parser


def finalize_args(args):
    """Normalize/patch args for backward compatibility."""
    if getattr(args, "epoch", None) is not None:
        args.epochs = args.epoch

    # alias: --group -> --group_name
    if getattr(args, 'group_name', None) is not None and getattr(args, 'group', None) is not None:
        args.group_name = args.group

    # For Mamba backbones, lr=1e-3 often diverges; use a safer default unless overridden.
    if getattr(args, 'pred_model', None) == 'mamba' and abs(float(args.pred_lr_init) - 1e-3) < 1e-12:
        args.pred_lr_init = 3e-4
    if getattr(args, 'recon_model', None) == 'mamba' and abs(float(args.ae_lr_init) - 1e-3) < 1e-12:
        args.ae_lr_init = 3e-4

    args.model = args.model + args.pred_model
    return args
