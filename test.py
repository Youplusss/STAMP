# -*- coding: utf-8 -*-

import os
import argparse

import torch

from lib.evaluate import get_final_result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='LLM-TSAD Testing (STAMP pipeline + optional Mamba/LLM prediction branch)')

    # -------------------------- basic --------------------------
    parser.add_argument('--data', type=str, default='SWaT',
                        help='dataset name (SWaT, WADI, SMD, SMAP, MSL, ...)')
    parser.add_argument('--group', type=str, default='machine-1-1',
                        help='SMD group name, e.g. machine-1-1')

    parser.add_argument('--debug', default=False, type=eval)
    parser.add_argument('--real_value', default=False, type=eval)
    parser.add_argument('--log_dir', default="expe", type=str)
    parser.add_argument('--model', default="v2_", type=str)

    parser.add_argument('--pred_model', default="mamba", type=str,
                        help='gat | mamba | llm(Time-LLM-style)')
    parser.add_argument('--recon_model', default="mamba", type=str,
                        help='ae | mamba')

    parser.add_argument('--gpu_id', default="0", type=str)
    parser.add_argument('--temp_method', default="SAttn", type=str)

    # SWaT/WADI 允许自定义路径
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)

    # -------------------------- graph / STAMP-GAT params --------------------------
    parser.add_argument('--nnodes', type=int, default=45)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--em_dim', type=int, default=32)
    parser.add_argument('--alpha', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--att_option', type=int, default=1)

    # -------------------------- attention params (STAMP-GAT) --------------------------
    # These are required by model.net.STATModel and exist in run.py; keep defaults consistent for compatibility.
    parser.add_argument('--embed_size', type=int, default=64, help='embed_size (for temporal/spatial attention in STAMP-GAT)')
    parser.add_argument('--num_heads', type=int, default=8, help='num_heads (for MultiheadAttention)')
    parser.add_argument('--num_layers', type=int, default=1, help='num_attn_layers')
    parser.add_argument('--ffwd_size', type=int, default=32, help='ffwd_size')
    parser.add_argument('--is_conv', type=eval, default=False, help='whether to use conv FFN in TransformerEncoder')
    parser.add_argument('--return_weight', type=eval, default=False, help='whether to return attention weights (if supported)')

    # -------------------------- sequence params --------------------------
    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--n_pred', type=int, default=3)
    parser.add_argument('--temp_kernel', type=int, default=5)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=1)

    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default="GLU")

    # -------------------------- AE (old) params --------------------------
    parser.add_argument('--latent_size', type=int, default=1)

    # -------------------------- runtime params --------------------------
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--val_ratio', type=float, default=.2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--test_alpha', type=float, default=.5)
    parser.add_argument('--test_beta', type=float, default=.5)
    parser.add_argument('--test_gamma', type=float, default=0.5)
    parser.add_argument('--test_topk', type=int, default=1, help='number of top-K nodes used to form final score')
    parser.add_argument('--test_topk_agg', type=str, default='sum', choices=['sum', 'mean', 'max'], help='how to aggregate the top-K node scores')
    parser.add_argument('--search_steps', default=50, type=int)

    parser.add_argument('--is_down_sample', type=eval, default=True)
    parser.add_argument('--down_len', type=int, default=100)

    parser.add_argument('--is_mas', default=True, type=eval)

    # -------------------------- Mamba Forecast params --------------------------
    parser.add_argument('--mamba_use_mas', default=True, type=eval)
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
    parser.add_argument('--recon_num_layers', type=str, default='2,2,2')
    parser.add_argument('--recon_d_state', type=int, default=16)
    parser.add_argument('--recon_d_conv', type=int, default=4)
    parser.add_argument('--recon_expand', type=int, default=2)
    parser.add_argument('--recon_dropout', type=float, default=0.1)
    parser.add_argument('--recon_output_activation', type=str, default='auto', help='auto | none | sigmoid | tanh')

    # -------------------------- Time-LLM (LLM forecaster) params --------------------------
    # Only used when --pred_model llm
    parser.add_argument('--llm_use_mas', default=False, type=eval,
                        help='(pred_model=llm) whether to fuse MAS channels as extra variables')
    parser.add_argument('--llm_backend', type=str, default='gpt2',
                        help='(pred_model=llm) backend type: gpt2 | bert | llama (auto via AutoModel)')
    parser.add_argument('--llm_model', type=str, default='gpt2',
                        help='(pred_model=llm) HuggingFace model name or local path')
    parser.add_argument('--llm_pretrained', default=True, type=eval,
                        help='(pred_model=llm) True: use pretrained weights; False: random init (ablation)')
    parser.add_argument('--llm_layers', type=int, default=6,
                        help='(pred_model=llm) truncate to first K transformer layers')
    parser.add_argument('--llm_dtype', type=str, default='auto', choices=['auto', 'float16', 'bfloat16', 'float32'],
                        help='(pred_model=llm) LLM weights dtype')
    parser.add_argument('--llm_grad_ckpt', default=False, type=eval,
                        help='(pred_model=llm) enable gradient checkpointing in LLM')
    parser.add_argument('--llm_load_in_4bit', default=False, type=eval,
                        help='(pred_model=llm) load LLM in 4-bit quantization (requires bitsandbytes)')
    parser.add_argument('--llm_load_in_8bit', default=False, type=eval,
                        help='(pred_model=llm) load LLM in 8-bit quantization (requires bitsandbytes)')

    # Patch / reprogramming
    parser.add_argument('--llm_patch_len', type=int, default=4)
    parser.add_argument('--llm_stride', type=int, default=2)
    parser.add_argument('--llm_d_model', type=int, default=32)
    parser.add_argument('--llm_d_ff', type=int, default=32)
    parser.add_argument('--llm_n_heads', type=int, default=4)
    parser.add_argument('--llm_dropout', type=float, default=0.1)
    parser.add_argument('--llm_head_dropout', type=float, default=0.0)
    parser.add_argument('--llm_pred_activation', type=str, default='none', choices=['none', 'sigmoid', 'tanh'])

    # Prompt
    parser.add_argument('--llm_prompt_mode', type=str, default='stats_short',
                        choices=['none', 'dataset', 'stats', 'stats_short'])
    parser.add_argument('--llm_prompt_root', type=str, default='expe/prompt_bank')
    parser.add_argument('--llm_prompt_domain', type=str, default='anomaly_detection')
    parser.add_argument('--llm_top_k_lags', type=int, default=5)

    # Optional cross-feature mixer
    parser.add_argument('--llm_feature_mixer', type=str, default='none', choices=['none', 'mlp'])
    parser.add_argument('--llm_mixer_rank', type=int, default=64)

    return parser


def load_dataloaders(args, device):
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
    try:
        x = train_loader.dataset.x
        N = int(x.shape[2])
        C = int(x.shape[3])
        if args.nnodes != N:
            print(f"[Info] Override args.nnodes: {args.nnodes} -> {N} (from data)")
            args.nnodes = N
        if args.in_channels != C:
            print(f"[Info] Override args.in_channels: {args.in_channels} -> {C} (from data)")
            args.in_channels = C
    except Exception as e:
        print(f"[Warn] Failed to infer nnodes/in_channels from dataset: {e}")


def main():
    args = build_arg_parser().parse_args()

    # 训练脚本会把 args.model 变成 v2_<pred_model>
    args.model = args.model + args.pred_model


    # Resolve experiment directories
    from lib.paths import resolve_experiment_dirs
    exp = resolve_experiment_dirs(args.log_dir)
    args.run_id = exp.run_id
    args.log_dir = exp.root
    args.log_dir_log = exp.log_dir
    args.log_dir_pth = exp.pth_dir
    args.log_dir_pdf = exp.pdf_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    from lib.utils import get_default_device, concate_results
    from lib.logger import get_logger, log_hparams
    from lib.logger import log_test_results
    from model.utils import init_seed

    # device
    device = get_default_device()
    args.device = device

    # load data
    train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = load_dataloaders(args, device)
    infer_and_override_data_shape(args, train_loader)

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

    init_seed(args.seed)

    pred_model_type = args.pred_model.lower()
    recon_model_type = args.recon_model.lower()

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

    # build pred
    if pred_model_type == 'mamba':
        pred_model = mamba_pred
    elif pred_model_type == 'llm':
        pred_model = llm_pred
    else:
        from model.net import STATModel
        channels_list = [[16, 8, 32], [32, 8, 64]]
        pred_model = STATModel(args, device, args.window_size - args.n_pred, channels_list, static_feat=None)

    # build recon
    if recon_model_type == 'mamba':
        ae_model = mamba_ae
    else:
        from model.net import EncoderDecoder
        AE_IN_CHANNELS = args.window_size * args.nnodes * args.in_channels
        latent_size = args.window_size * args.latent_size
        ae_model = EncoderDecoder(AE_IN_CHANNELS, latent_size, AE_IN_CHANNELS, not args.real_value)

    # logger
    logger = get_logger(exp.log_dir, name=args.model, debug=args.debug, data=args.data, tag='test', model=args.model, run_id=exp.run_id, console=True)
    log_hparams(logger, args)

    model_path = os.path.join(exp.pth_dir, 'best_model_' + args.data + "_" + args.model + '.pth')
    print("load model:", model_path)

    from trainer import Tester
    tester = Tester(
        pred_model,
        ae_model,
        args,
        min_max_scaler,
        logger,
        path=model_path,
        alpha=args.test_alpha,
        beta=args.test_beta,
        gamma=args.test_gamma,
    )

    map_location = torch.device(device)

    test_results = tester.testing(test_loader, map_location)

    (test_y_pred,
     test_loss1_list,
     test_loss2_list,
     test_pred_list,
     test_gt_list,
     test_origin_list,
     test_construct_list,
     test_generate_list,
     test_generate_construct_list) = concate_results(test_results)

    print("scores:", len(test_y_pred), float(test_y_pred.mean()))
    print("loss1:", len(test_loss1_list), float(test_loss1_list.mean()))
    print("loss2:", len(test_loss2_list), float(test_loss2_list.mean()))
    print("y_test_labels:", len(y_test_labels))

    # write summary into the test log
    print(f"[Test Weights] alpha={args.test_alpha} beta={args.test_beta} gamma={args.test_gamma}")

    score_mean = float(test_y_pred.mean())
    loss1_mean = float(test_loss1_list.mean())
    loss2_mean = float(test_loss2_list.mean())

    test_pred_results = [test_pred_list, test_gt_list]
    test_ae_results = [test_construct_list, test_origin_list]
    test_generate_results = [test_generate_list, test_generate_construct_list]

    best_by_method = {}

    # search best f1
    print("================= Find best f1 from score (method=max) =================")
    info, test_scores, predict = get_final_result(
        test_pred_results,
        test_ae_results,
        test_generate_results,
        y_test_labels,
        option=2,
        method="max",
        topk=args.test_topk, topk_agg=args.test_topk_agg, alpha=args.test_alpha,
        beta=args.test_beta,
        gamma=args.test_gamma,
        search_steps=args.search_steps,
    )
    print(info)
    best_by_method['max'] = info

    print("\n================= Find best f1 from score (method=sum) =================")
    info, test_scores, predict = get_final_result(
        test_pred_results,
        test_ae_results,
        test_generate_results,
        y_test_labels,
        option=2,
        method="sum",
        topk=args.test_topk, topk_agg=args.test_topk_agg, alpha=args.test_alpha,
        beta=args.test_beta,
        gamma=args.test_gamma,
        search_steps=args.search_steps,
    )
    print(info)
    best_by_method['sum'] = info

    print("\n================= Find best f1 from score (method=mean) =================")
    info, test_scores, predict = get_final_result(
        test_pred_results,
        test_ae_results,
        test_generate_results,
        y_test_labels,
        option=2,
        method="mean",
        topk=args.test_topk, topk_agg=args.test_topk_agg, alpha=args.test_alpha,
        beta=args.test_beta,
        gamma=args.test_gamma,
        search_steps=args.search_steps,
    )
    print(info)
    best_by_method['mean'] = info

    log_test_results(
        logger,
        dataset=args.data,
        model=args.model,
        checkpoint_path=model_path,
        score_mean=score_mean,
        loss1_mean=loss1_mean,
        loss2_mean=loss2_mean,
        best_by_method=best_by_method,
    )


if __name__ == '__main__':
    main()
