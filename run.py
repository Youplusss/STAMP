# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn


def _parse_use_adv(v):
    """Parse --use_adv flag.

    Supports:
    - 0/1/2 (recommended)
    - True/False (backward compatible): True -> 2 (use current/stable adv), False -> 0
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='LLM-TSAD Training (STAMP pipeline + optional Mamba/LLM prediction branch)')

    # -------------------------- basic --------------------------
    parser.add_argument('--data', type=str, default='SWaT',
                        help='dataset name (SWaT, WADI, SMD, SMAP, MSL, ...)')
    # gpu selection (kept for compatibility with test.py/lib/cli.py)
    # Note: this is a string because CUDA_VISIBLE_DEVICES accepts comma-separated ids like "0,1".
    parser.add_argument('--gpu_id', default="0", type=str,
                        help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1"')
    # optional explicit dataset files (override defaults under dataset/{data}/...)
    parser.add_argument('--train_file', default=None, type=str,
                        help='(optional) path to training CSV/PKL; if not set, use dataset defaults')
    parser.add_argument('--test_file', default=None, type=str,
                        help='(optional) path to test CSV/PKL; if not set, use dataset defaults')
    # optional group name for SMD (when using generated CSV groups)
    parser.add_argument('--group', default=None, type=str,
                        help='(optional) group name for SMD CSV mode (used by lib.paths.resolve_dataset_paths)')
    parser.add_argument('--debug', default=False, type=eval)
    parser.add_argument('--real_value', default=False, type=eval)
    parser.add_argument('--log_dir', default="expe", type=str)
    # 兼容 Trainer.transfer_path（老代码会用到 log_dir_transfer）
    parser.add_argument('--log_dir_transfer', default=None, type=str,
                        help='(optional) transfer checkpoint dir; default=log_dir')

    parser.add_argument('--model', default="v2_", type=str)
    # pred_model: gat | mamba | llm
    parser.add_argument('--pred_model', default="mamba", type=str,
                        help="prediction branch type: gat | mamba | llm(Time-LLM-style)")
    # recon_model: ae | mamba
    parser.add_argument('--recon_model', default="mamba", type=str,
                        help="reconstruction branch type: ae | mamba")

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

    parser.add_argument('--is_down_sample', type=eval, default=True, help='down-sample raw series or not')
    parser.add_argument('--down_len', type=int, default=100, help='down sample ratio')

    # 对抗/耦合训练中是否冻结另一分支参数（推荐 True，显存/速度更友好）
    parser.add_argument('--adv_freeze_other', type=eval, default=True)

    # -------------------------- adversarial / coupled training (NEW) --------------------------
    # 为了调试与适配 Mamba，本 repo 的 Trainer 支持关闭对抗训练，以及多种更稳定的损失形式。
    parser.add_argument(
        '--use_adv',
        type=_parse_use_adv,
        default=2,
        help='training loss mode: 0=no adversarial (weighted sum baseline), 1=legacy4 (tmp/trainer.py original), 2=current stabilized design. '
             'Also accepts True/False for backward compatibility (True->2, False->0).'
    )

    # baseline (use_adv=0) weights
    parser.add_argument('--loss_weight_pred', type=float, default=1.0,
                        help='(use_adv=0) weight for prediction loss')
    parser.add_argument('--loss_weight_ae', type=float, default=1.0,
                        help='(use_adv=0) weight for reconstruction loss')

    parser.add_argument('--adv_train_strategy', type=str, default='legacy4',
                        choices=['legacy4', '4step', '2step'],
                        help='legacy4: original 4 updates per batch + 5/e,3/e schedule; '
                             '4step: 4 updates but stabilized objectives; '
                             '2step: 2 updates (GAN-style), recommended for Mamba.')
    parser.add_argument('--adv_scope', type=str, default='full',
                        choices=['full', 'pred', 'history'],
                        help='where to compute adversarial reconstruction loss on generated window: '
                             'full=whole window; pred=only last n_pred steps (recommended); history=only context part.')
    parser.add_argument('--adv_mode', type=str, default='legacy',
                        choices=['legacy', 'hinge', 'band', 'softplus', 'softplus0', 'exp', 'began'],
                        help='AE adversarial objective. legacy: ae_loss - lambda*adv_loss (can be unstable). '
                             'hinge/softplus0/softplus/exp are bounded variants (recommended for Mamba). began is a BEGAN-style equilibrium variant.')
    parser.add_argument('--adv_margin', type=float, default=0.1,
                        help='margin value used by hinge/softplus modes. See trainer.py/lib/adv_losses.py for definition.')
    parser.add_argument('--adv_margin_high', type=float, default=-1.0,
                        help='upper margin for adv_mode=band (two-sided hinge). If <=0, band mode is invalid.')
    parser.add_argument('--adv_margin_mode', type=str, default='rel', choices=['abs', 'rel'],
                        help='margin type: abs => adv_loss >= ae_loss + margin; rel => adv_loss >= (1+margin)*ae_loss (scale-invariant).')
    parser.add_argument('--adv_tau', type=float, default=1.0,
                        help='temperature for softplus/exp modes (smoother gradients).')

    # --- extra stabilizers / knobs (v2) ---
    parser.add_argument('--adv_pred_objective', type=str, default='adv',
                        choices=['adv', 'gap_relu', 'gap_hinge'],
                        help='how the pred branch uses the AE signal. adv: add adv_loss; '
                             'gap_relu: penalize positive (adv-ae); gap_hinge: penalize (adv-ae-margin) when too large.')
    parser.add_argument('--adv_margin_floor', type=float, default=0.0,
                        help='a minimum absolute margin added on top of rel margin (helps when ae_loss is tiny).')
    parser.add_argument('--adv_tau_mode', type=str, default='abs', choices=['abs', 'rel'],
                        help='temperature scaling for softplus/softplus0/exp: abs=tau, rel=tau*ae_loss_detached (scale-invariant).')
    parser.add_argument('--adv_tau_floor', type=float, default=1e-4,
                        help='minimum temperature value to avoid numerical issues when tau_mode=rel and ae_loss is tiny.')
    parser.add_argument('--adv_energy_transform', type=str, default='none', choices=['none', 'log1p', 'sqrt'],
                        help='apply a transform to reconstruction energies before adversarial comparisons (stabilizes large adv_loss).')
    parser.add_argument('--adv_auto_balance', type=eval, default=False,
                        help='auto-rescale adversarial weights so adv terms do not dominate base losses (recommended if you see divergence).')

    # BEGAN-style equilibrium (only used when --adv_mode began)
    parser.add_argument('--adv_began_gamma', type=float, default=0.5,
                        help='target ratio for BEGAN equilibrium: E_fake ~= gamma * E_real (on transformed energy).')
    parser.add_argument('--adv_began_lambda_k', type=float, default=0.001,
                        help='update rate for BEGAN k. Smaller => more stable.')
    parser.add_argument('--adv_began_k_init', type=float, default=0.0,
                        help='initial k for BEGAN.')

    # weight schedule (warmup + linear ramp) for stabilized strategies
    parser.add_argument('--adv_lambda_pred', type=float, default=1.0,
                        help='max weight for adv_loss in pred update (after warmup+ramp).')
    parser.add_argument('--adv_lambda_ae', type=float, default=1.0,
                        help='max weight for adversarial penalty in AE update (after warmup+ramp).')
    parser.add_argument('--adv_warmup_epochs', type=int, default=1,
                        help='warmup epochs before enabling adversarial weights.')
    parser.add_argument('--adv_ramp_epochs', type=int, default=5,
                        help='ramp epochs to linearly increase adversarial weights after warmup.')

    # optional weight decay knobs (Adam) (kept consistent with lib/cli.py)
    parser.add_argument('--pred_weight_decay', type=float, default=1e-4,
                        help='weight decay for prediction model optimizer')
    parser.add_argument('--ae_weight_decay', type=float, default=1e-4,
                        help='weight decay for reconstruction model optimizer')

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

    # -------------------------- Time-LLM (LLM forecaster) params --------------------------
    # Only used when --pred_model llm
    parser.add_argument('--llm_use_mas', default=False, type=eval,
                        help='(pred_model=llm) whether to fuse MAS channels as extra variables')
    parser.add_argument('--llm_backend', type=str, default='gpt2',
                        help='(pred_model=llm) backend type: gpt2 | bert | llama (auto via AutoModel)')
    parser.add_argument('--llm_model', type=str, default='gpt2',
                        help='(pred_model=llm) HuggingFace model name or local path (e.g., gpt2, bert-base-uncased, meta-llama/...)')
    # HuggingFace loading controls
    parser.add_argument('--hf_cache_dir', type=str, default=None,
                        help='(pred_model=llm) optional HuggingFace cache_dir passed to from_pretrained (e.g., /home/xxx/code/huggingface)')
    parser.add_argument('--hf_local_files_only', default=False, type=eval,
                        help='(pred_model=llm) if True, do not download; only load local files (offline-safe).')

    parser.add_argument('--llm_pretrained', default=True, type=eval,
                        help='(pred_model=llm) True: use pretrained weights; False: random init (ablation)')
    parser.add_argument('--llm_layers', type=int, default=6,
                        help='(pred_model=llm) truncate to first K transformer layers (memory/speed control)')
    parser.add_argument('--llm_dtype', type=str, default='auto', choices=['auto', 'float16', 'bfloat16', 'float32'],
                        help='(pred_model=llm) LLM weights dtype (auto uses HF default)')
    parser.add_argument('--llm_grad_ckpt', default=False, type=eval,
                        help='(pred_model=llm) enable gradient checkpointing in LLM (reduces memory, slower)')
    parser.add_argument('--llm_load_in_4bit', default=False, type=eval,
                        help='(pred_model=llm) load LLM in 4-bit quantization (requires bitsandbytes)')
    parser.add_argument('--llm_load_in_8bit', default=False, type=eval,
                        help='(pred_model=llm) load LLM in 8-bit quantization (requires bitsandbytes)')

    # KV-cache: should be disabled for training to reduce memory
    parser.add_argument('--llm_use_cache', default=False, type=eval,
                        help='(pred_model=llm) pass use_cache to the HF model forward. For training this should be False to save memory.')

    # Patch / reprogramming
    parser.add_argument('--llm_patch_len', type=int, default=4, help='(pred_model=llm) patch length (must be <= context_len)')
    parser.add_argument('--llm_stride', type=int, default=2, help='(pred_model=llm) patch stride')
    parser.add_argument('--llm_d_model', type=int, default=32, help='(pred_model=llm) patch embedding dim')
    parser.add_argument('--llm_d_ff', type=int, default=32, help='(pred_model=llm) bottleneck dim used from LLM hidden states')
    parser.add_argument('--llm_n_heads', type=int, default=4, help='(pred_model=llm) reprogramming attention heads')
    parser.add_argument('--llm_dropout', type=float, default=0.1, help='(pred_model=llm) dropout in patch/reprogramming')
    parser.add_argument('--llm_head_dropout', type=float, default=0.0, help='(pred_model=llm) dropout in output head')
    parser.add_argument('--llm_pred_activation', type=str, default='none', choices=['none', 'sigmoid', 'tanh'],
                        help='(pred_model=llm) optional activation on predictions')

    # Prompt
    parser.add_argument('--llm_prompt_mode', type=str, default='stats_short',
                        choices=['none', 'dataset', 'stats', 'stats_short'],
                        help='(pred_model=llm) prompt ablation: none/dataset/stats/stats_short')
    parser.add_argument('--llm_prompt_root', type=str, default='expe/prompt_bank',
                        help='(pred_model=llm) prompt bank root dir')
    parser.add_argument('--llm_prompt_domain', type=str, default='anomaly_detection',
                        help='(pred_model=llm) prompt domain string (for future extensions)')
    parser.add_argument('--llm_top_k_lags', type=int, default=5,
                        help='(pred_model=llm) top-k FFT lags inserted into prompt (0 disables)')

    # Optional cross-feature mixer (helps when N is large and correlations matter)
    parser.add_argument('--llm_feature_mixer', type=str, default='none', choices=['none', 'mlp'],
                        help='(pred_model=llm) lightweight cross-feature mixer before patching. none | mlp(low-rank)')
    parser.add_argument('--llm_mixer_rank', type=int, default=64,
                        help='(pred_model=llm) low-rank dimension for --llm_feature_mixer mlp')

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

        train_file = getattr(args, 'train_file', None) or default_train
        test_file = getattr(args, 'test_file', None) or default_test

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

    # 名称：v2_mamba / v2_gat...
    args.model = args.model + args.pred_model


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Resolve experiment directories (expe/log, expe/pth, expe/pdf)
    from lib.paths import resolve_experiment_dirs
    exp = resolve_experiment_dirs(args.log_dir)
    args.run_id = exp.run_id
    args.log_dir = exp.root
    args.log_dir_log = exp.log_dir
    args.log_dir_pth = exp.pth_dir
    args.log_dir_pdf = exp.pdf_dir
    # keep transfer dir consistent unless user overrides
    if getattr(args, 'log_dir_transfer', None) in (None, '', 'None'):
        args.log_dir_transfer = exp.pth_dir

    from lib.utils import get_default_device, to_device, plot_history, plot_history2
    from lib.metrics import masked_mse_loss
    from model.utils import print_model_parameters, init_seed

    # logger (write into expe/log)
    from lib.logger import get_logger, log_hparams
    logger = get_logger(exp.log_dir, name=args.model, debug=args.debug, data=args.data, tag='train', model=args.model, run_id=exp.run_id, console=True)
    log_hparams(logger, args)

    # device
    print(torch.cuda.is_available())
    device = get_default_device()
    # make device accessible to Trainer/Tester
    args.device = device

    # load data
    train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = load_dataloaders(args, device)

    # infer shape
    infer_and_override_data_shape(args, train_loader)

    # Ensure output dirs exist even in remote/nohup runs
    os.makedirs(exp.log_dir, exist_ok=True)
    os.makedirs(exp.pth_dir, exist_ok=True)
    os.makedirs(exp.pdf_dir, exist_ok=True)

    if args.in_channels != args.out_channels:
        msg = (
            "For STAMP-style generate_batch concat, require in_channels==out_channels, "
            + "got "
            + str(args.in_channels)
            + " vs "
            + str(args.out_channels)
            + "."
        )
        raise ValueError(msg)

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

    # LLM prediction branch (Time-LLM style)
    if pred_model_type == 'llm':
        from model.llm_wrappers import build_stamp_llm_predictor
        llm_pred = build_stamp_llm_predictor(args)
    else:
        llm_pred = None

    # pred branch
    if pred_model_type == 'mamba':
        pred_model = mamba_pred
    elif pred_model_type == 'llm':
        pred_model = llm_pred
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
    # Optimizers (keep weight_decay configurable; some Mamba backbones are sensitive)
    pred_wd = float(getattr(args, 'pred_weight_decay', 1e-4))
    ae_wd = float(getattr(args, 'ae_weight_decay', 1e-4))
    pred_params = list(pred_model.parameters())
    ae_params = list(ae_model.parameters())
    if len(pred_params) == 0:
        raise ValueError('Prediction model has no trainable parameters. (rollback version requires both branches enabled)')
    if len(ae_params) == 0:
        raise ValueError('Reconstruction model has no trainable parameters. (rollback version requires both branches enabled)')
    pred_optimizer = torch.optim.Adam(pred_params, lr=args.pred_lr_init, eps=1.0e-8, weight_decay=pred_wd)
    ae_optimizer = torch.optim.Adam(ae_params, lr=args.ae_lr_init, eps=1.0e-8, weight_decay=ae_wd)


    pred_loss = masked_mse_loss(mask_value=-0.01)
    ae_loss = masked_mse_loss(mask_value=-0.01)

    # print params
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

    # attach unified logger
    trainer.logger = logger

    train_history, val_history = trainer.train()

    plot_history(train_history, model=args.model, mode="train", data=args.data, out_dir=exp.pdf_dir, show=False)
    plot_history(val_history, model=args.model, mode="val", data=args.data, out_dir=exp.pdf_dir, show=False)
    plot_history2(val_history, model=args.model, mode="val", data=args.data, out_dir=exp.pdf_dir, show=False)


if __name__ == '__main__':
    main()
