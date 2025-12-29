import argparse


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments shared by train/test and sup/unsup entrypoints."""

    # Dataset + paths
    parser.add_argument('--data', type=str, default='SWaT', help='dataset name (SWaT, WADI, SMD, MSL, SMAP, ...)')
    parser.add_argument('--data_root', type=str, default=None, help='override repo ./data directory')
    parser.add_argument('--dataset_root', type=str, default=None, help='override repo ./dataset directory (csv datasets)')
    parser.add_argument('--train_file', type=str, default=None, help='override train file path (csv datasets)')
    parser.add_argument('--test_file', type=str, default=None, help='override test file path (csv datasets)')
    parser.add_argument('--group_name', type=str, default=None, help='SMD subset name, e.g. machine-1-1')

    # Runtime
    parser.add_argument('--debug', default=False, type=eval)
    parser.add_argument('--real_value', default=False, type=eval)
    parser.add_argument('--log_dir', default="expe", type=str)
    parser.add_argument('--gpu_id', default="0", type=str)

    # Model choice
    parser.add_argument('--model', default="v2_", type=str)
    parser.add_argument('--pred_model', default="gat", type=str)
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
    parser.add_argument('--test_beta', type=float, default=.0)
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

    return parser


def finalize_args(args):
    """Normalize/patch args for backward compatibility."""
    if getattr(args, "epoch", None) is not None:
        args.epochs = args.epoch
    args.model = args.model + args.pred_model
    return args

