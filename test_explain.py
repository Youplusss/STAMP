# -*- coding: utf-8 -*-

import os
import argparse
import sys

import torch

from lib.evaluate import get_final_result
from test import load_dataloaders, infer_and_override_data_shape


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='STAMP Testing (支持 Mamba 预测/重构分支)')

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
                        help='gat | mamba')
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
    parser.add_argument('--test_beta', type=float, default=.0)
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

    # -------------------------- LLM explanation (optional) --------------------------
    parser.add_argument('--do_explain', default=False, type=eval,
                        help='whether to generate LLM/template explanations for detected anomalies')
    parser.add_argument('--explain_use_method', type=str, default='best',
                        choices=['best', 'max', 'sum', 'mean'],
                        help='which scoring aggregation method to use for explanation (must match get_final_result method)')
    parser.add_argument('--explain_backend', type=str, default='template',
                        choices=['template', 'hf'],
                        help='explanation backend: template (no deps) | hf (HuggingFace transformers causal LM)')
    parser.add_argument('--explain_llm_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='HuggingFace model name or local path (only used when explain_backend=hf). Default: Qwen2.5-7B-Instruct.')
    parser.add_argument('--explain_cuda_visible_devices', type=str, default=None,
                        help='Optional: set CUDA_VISIBLE_DEVICES for the explanation LLM process (e.g., "3"). This is useful to pin the LLM to one GPU.')
    parser.add_argument('--explain_language', type=str, default='zh',
                        help='prompt language: zh or en')
    parser.add_argument('--explain_topk_features', type=int, default=5,
                        help='number of top contributing variables to include in evidence')
    parser.add_argument('--explain_max_segments', type=int, default=20,
                        help='maximum number of anomaly segments to explain')
    parser.add_argument('--explain_max_new_tokens', type=int, default=256,
                        help='LLM generation max_new_tokens')
    parser.add_argument('--explain_temperature', type=float, default=0.0,
                        help='LLM generation temperature (0 for deterministic)')
    parser.add_argument('--explain_do_sample', default=False, type=eval,
                        help='LLM generation do_sample')
    parser.add_argument('--explain_top_p', type=float, default=0.95,
                        help='LLM generation top_p')
    parser.add_argument('--explain_repetition_penalty', type=float, default=1.05,
                        help='LLM generation repetition_penalty')
    parser.add_argument('--explain_out_json', type=str, default=None,
                        help='output path for explanations.json (default: <log_dir>/explanations_<data>_<method>.json)')
    parser.add_argument('--explain_out_md', type=str, default=None,
                        help='output path for explanations.md (default: <log_dir>/explanations_<data>_<method>.md)')
    parser.add_argument('--explain_trust_remote_code', default=False, type=eval,
                        help='(hf backend) trust_remote_code when loading the model')
    parser.add_argument('--explain_local_files_only', default=False, type=eval,
                        help='(hf backend) only load local files (no auto-download)')
    parser.add_argument('--explain_hf_endpoint', type=str, default="https://hf-mirror.com",
                        help='(hf backend) override HuggingFace Hub endpoint, e.g. https://hf-mirror.com (otherwise uses env HF_ENDPOINT)')
    parser.add_argument('--explain_hf_cache_dir', type=str, default='/home/youwenlong/code/huggingface',
                        help=(
                            '(hf backend) cache dir. Default: /home/youwenlong/code/huggingface. '
                            'We will create the directory if it does not exist. '
                            'You can override it with --explain_hf_cache_dir or by setting env HF_HOME/TRANSFORMERS_CACHE.'
                        ))
    parser.add_argument('--explain_hf_token', type=str, default=None,
                        help='(hf backend) HuggingFace access token. Helps avoid 429 rate limits on mirrors/HF Hub. You can also set env HF_TOKEN/HUGGINGFACE_HUB_TOKEN.')
    parser.add_argument('--explain_force_gpu', default=False, type=eval,
                        help='(hf backend) force running the LLM on CUDA. Default False because some torch+transformers+CUDA combos crash with device-side assert.')
    parser.add_argument('--explain_all_methods', default=False, type=eval,
                        help='Generate explanations for all aggregation methods (max/sum/mean) instead of only the chosen method.')
    parser.add_argument('--explain_hf_load_in_4bit', default=True, type=eval,
                        help='(hf backend) Load LLM in 4-bit (bitsandbytes) to reduce VRAM. Recommended for 7B on ~16GB VRAM.')
    parser.add_argument('--explain_hf_load_in_8bit', default=False, type=eval,
                        help='(hf backend) Load LLM in 8-bit (bitsandbytes). Use if 4-bit is not desired.')
    parser.add_argument('--explain_progress', default=True, type=eval,
                        help='Show a tqdm progress bar during LLM explanation generation (per anomaly segment).')
    parser.add_argument('--explain_log_progress_every', type=int, default=1,
                        help='When not running in a TTY (e.g., nohup redirect), print a log-friendly progress line every N segments (1=every segment).')

    return parser


def main():
    args = build_arg_parser().parse_args()

    # Make stdout/stderr line-buffered under nohup so `tail -f` shows incremental progress.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Optional: pin GPUs for this process.
    # IMPORTANT: do NOT override an already-set CUDA_VISIBLE_DEVICES unless user explicitly requests it.
    if getattr(args, 'explain_cuda_visible_devices', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.explain_cuda_visible_devices)
    else:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu_id))

    # Optional: set HF token for authenticated hub access (helps avoid 429 rate limits)
    if getattr(args, 'explain_hf_token', None):
        tok = str(args.explain_hf_token).strip()
        if tok:
            os.environ['HF_TOKEN'] = tok
            os.environ['HUGGINGFACE_HUB_TOKEN'] = tok

    # Normalize user paths early (important for HF cache dir like "~/.cache/huggingface")
    if getattr(args, 'explain_hf_cache_dir', None):
        args.explain_hf_cache_dir = os.path.expanduser(str(args.explain_hf_cache_dir))
        os.makedirs(args.explain_hf_cache_dir, exist_ok=True)
        # Export cache envs so huggingface_hub/transformers use a single consistent cache.
        os.environ["HF_HOME"] = args.explain_hf_cache_dir
        os.environ.setdefault("TRANSFORMERS_CACHE", args.explain_hf_cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(args.explain_hf_cache_dir, "hub"))

    # NOTE: args.model is part of the checkpoint filename suffix.
    if args.model in ("v2_", "v2", "v2-"):
        args.model = "v2_" + str(args.pred_model)

    # Resolve experiment directories
    from lib.paths import resolve_experiment_dirs
    exp = resolve_experiment_dirs(args.log_dir)
    args.run_id = exp.run_id
    args.log_dir = exp.root
    args.log_dir_log = exp.log_dir
    args.log_dir_pth = exp.pth_dir
    args.log_dir_pdf = exp.pdf_dir


    from lib.utils import get_default_device, concate_results
    from model.utils import init_seed

    # device
    args.device = get_default_device()

    # logger
    from lib.logger import get_logger, log_hparams
    logger = get_logger(exp.log_dir, name=args.model, debug=args.debug, data=args.data, tag='test', model=args.model, run_id=exp.run_id, console=True)
    log_hparams(logger, args)

    # checkpoint path
    model_path = os.path.join(exp.pth_dir, 'best_model_' + args.data + "_" + args.model + '.pth')
    print("load model:", model_path)

    # load data
    train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = load_dataloaders(args, args.device)
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

    # build pred
    if pred_model_type == 'mamba':
        pred_model = mamba_pred
    else:
        from model.net import STATModel
        channels_list = [[16, 8, 32], [32, 8, 64]]
        pred_model = STATModel(args, args.device, args.window_size - args.n_pred, channels_list, static_feat=None)

    # build recon
    if recon_model_type == 'mamba':
        ae_model = mamba_ae
    else:
        from model.net import EncoderDecoder
        AE_IN_CHANNELS = args.window_size * args.nnodes * args.in_channels
        latent_size = args.window_size * args.latent_size
        ae_model = EncoderDecoder(AE_IN_CHANNELS, latent_size, AE_IN_CHANNELS, not args.real_value)


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

    map_location = torch.device(args.device)

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

    test_pred_results = [test_pred_list, test_gt_list]
    test_ae_results = [test_construct_list, test_origin_list]
    test_generate_results = [test_generate_list, test_generate_construct_list]


    # search best f1 under different score aggregation methods
    results_by_method = {}

    for _method in ["max", "sum", "mean"]:
        print(f"\n================= Find best f1 from score (method={_method}) =================")
        info_m, test_scores_m, predict_m = get_final_result(
            test_pred_results,
            test_ae_results,
            test_generate_results,
            y_test_labels,
            option=2,
            method=_method,
            topk=args.test_topk, topk_agg=args.test_topk_agg,
            alpha=args.test_alpha, beta=args.test_beta, gamma=args.test_gamma,
            search_steps=args.search_steps,
        )
        print(info_m)
        results_by_method[_method] = {
            'info': info_m,
            'scores': test_scores_m,
            'predict': predict_m,
        }

    # optional: generate explanations
    if args.do_explain:
        use = str(args.explain_use_method).lower()
        if use == 'best':
            best_method = max(
                results_by_method.keys(),
                key=lambda k: float(results_by_method[k]['info'].get('best-f1', -1.0))
            )
        else:
            best_method = use
            if best_method not in results_by_method:
                raise ValueError(f"Unknown explain_use_method={use}; must be one of {list(results_by_method.keys())} or 'best'.")

        chosen = results_by_method[best_method]
        chosen_th = float(chosen['info'].get('threshold', 0.0))

        if args.explain_all_methods:
            for method_name, result in results_by_method.items():
                out_json = args.explain_out_json or os.path.join(args.log_dir, f"explanations_{args.data}_{method_name}.json")
                out_md = args.explain_out_md or os.path.join(args.log_dir, f"explanations_{args.data}_{method_name}.md")

                method_th = float(result['info'].get('threshold', chosen_th))
                print(f"\n[Explain] Using method={method_name}, threshold={method_th:.6f}")
                print(f"[Explain] Writing JSON to: {out_json}")
                print(f"[Explain] Writing MD   to: {out_md}")

                from explain.pipeline import generate_explanations
                _ = generate_explanations(
                    args=args,
                    dataset=args.data,
                    test_scores=result['scores'],
                    predict=result['predict'],
                    test_pred_results=test_pred_results,
                    test_ae_results=test_ae_results,
                    test_generate_results=test_generate_results,
                    option=2,
                    method=method_name,
                    threshold=method_th,
                    out_json_path=out_json,
                    out_md_path=out_md,
                )

                print(f"[Explain] Done. Explained segments: {_.get('num_segments', 0)}")
        else:
            out_json = args.explain_out_json or os.path.join(args.log_dir, f"explanations_{args.data}_{best_method}.json")
            out_md = args.explain_out_md or os.path.join(args.log_dir, f"explanations_{args.data}_{best_method}.md")

            print(f"\n[Explain] Using method={best_method}, threshold={chosen_th:.6f}")
            print(f"[Explain] Writing JSON to: {out_json}")
            print(f"[Explain] Writing MD   to: {out_md}")

            from explain.pipeline import generate_explanations
            _ = generate_explanations(
                args=args,
                dataset=args.data,
                test_scores=chosen['scores'],
                predict=chosen['predict'],
                test_pred_results=test_pred_results,
                test_ae_results=test_ae_results,
                test_generate_results=test_generate_results,
                option=2,
                method=best_method,
                threshold=chosen_th,
                out_json_path=out_json,
                out_md_path=out_md,
            )

            print(f"[Explain] Done. Explained segments: {_.get('num_segments', 0)}")


if __name__ == '__main__':
    main()
