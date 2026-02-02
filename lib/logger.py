import os
import logging
from datetime import datetime
from typing import Optional
import json


class TqdmConsoleHandler(logging.StreamHandler):
    """Console handler compatible with tqdm progress bars."""

    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            super().emit(record)


def _format_kv_block(title: str, items: list[tuple[str, object]]) -> str:
    lines = [f"[{title}]"]
    for k, v in items:
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def log_hparams(logger: logging.Logger, args, keys: Optional[list[str]] = None) -> None:
    """Log a detailed, prioritized hyperparameter summary.

    Design goals:
    - Put frequently-changed knobs up front (data/model/lr/batch/epochs/adv settings).
    - Put rarely-changed defaults later (dimensions, architecture defs).
    - Keep output stable and readable.
    """

    try:
        d = vars(args)
    except Exception:
        d = {}

    def pick(wanted: list[str]) -> list[tuple[str, object]]:
        return [(k, d.get(k)) for k in wanted if k in d]

    # Priority groups (top -> bottom)
    group_core = [
        'data', 'group',
        'model', 'pred_model', 'recon_model',
        'gpu_id', 'seed',
    ]

    group_train = [
        'epochs', 'batch_size',
        'pred_lr_init', 'ae_lr_init',
        'pred_weight_decay', 'ae_weight_decay',
        'dropout',
        'val_ratio',
        'grad_clip', 'max_grad_norm',
        'early_stop', 'early_stop_patience',
        'lr_decay', 'lr_decay_rate', 'lr_decay_step',
    ]

    group_data = [
        'is_down_sample', 'down_len',
        'is_mas', 'mamba_use_mas',
        'train_file', 'test_file',
        'data_root', 'dataset_root',
    ]

    group_adv = [
        'use_adv', 'adv_train_strategy', 'adv_freeze_other',
        'adv_scope',
        'adv_mode',
        'adv_margin_mode', 'adv_margin',
        'adv_tau',
        'adv_lambda_pred', 'adv_lambda_ae',
        'adv_warmup_epochs', 'adv_ramp_epochs',
    ]

    group_model_common = [
        'window_size', 'n_pred',
        'nnodes',
        'in_channels', 'out_channels',
    ]

    group_mamba = [
        'mamba_d_model', 'mamba_e_layers', 'mamba_d_state', 'mamba_d_conv', 'mamba_expand',
        'mamba_dropout', 'mamba_use_norm', 'mamba_use_last_residual',
        'recon_d_model', 'recon_num_layers', 'recon_d_state', 'recon_d_conv', 'recon_expand',
        'recon_dropout', 'recon_output_activation',
    ]

    group_gat = [
        'top_k', 'em_dim', 'alpha', 'hidden_dim', 'att_option',
        'embed_size', 'num_heads', 'num_layers', 'ffwd_size', 'is_conv', 'return_weight',
        'layer_num', 'act_func',
        'latent_size',
    ]

    # If caller explicitly passes keys, keep previous behavior (single list) but still add a title.
    if keys is not None:
        items = [(k, d.get(k)) for k in keys if k in d]
        logger.info(_format_kv_block('Hyperparameters', items))
        return

    blocks = [
        _format_kv_block('Run', pick(group_core)),
        _format_kv_block('Training', pick(group_train)),
        _format_kv_block('Data', pick(group_data)),
        _format_kv_block('Adversarial', pick(group_adv)),
        _format_kv_block('Shapes', pick(group_model_common)),
    ]

    # Only show the relevant backbone group(s)
    if str(d.get('pred_model', '')).lower() == 'mamba' or str(d.get('recon_model', '')).lower() == 'mamba':
        blocks.append(_format_kv_block('Mamba', pick(group_mamba)))
    if str(d.get('pred_model', '')).lower() == 'gat' or str(d.get('recon_model', '')).lower() == 'ae':
        blocks.append(_format_kv_block('GAT/AE', pick(group_gat)))

    logger.info("\n\n".join(blocks))


def log_test_results(
    logger: logging.Logger,
    *,
    dataset: str,
    model: str,
    checkpoint_path: str,
    score_mean: float | None = None,
    loss1_mean: float | None = None,
    loss2_mean: float | None = None,
    best_by_method: dict[str, dict] | None = None,
) -> None:
    """Write a structured test summary into the test log."""

    logger.info(_format_kv_block('TestRun', [
        ('data', dataset),
        ('model', model),
        ('checkpoint', checkpoint_path),
    ]))

    if score_mean is not None or loss1_mean is not None or loss2_mean is not None:
        logger.info(_format_kv_block('TestStats', [
            ('score_mean', score_mean),
            ('loss1_mean', loss1_mean),
            ('loss2_mean', loss2_mean),
        ]))

    if best_by_method:
        # best_by_method[method] = {'best-f1':..., 'precision':..., ...}
        for method, info in best_by_method.items():
            # keep a stable order for readability
            ordered_keys = ['best-f1', 'precision', 'recall', 'TP', 'TN', 'FP', 'FN', 'latency', 'threshold']
            items = [(k, info.get(k)) for k in ordered_keys if k in info]
            logger.info(_format_kv_block(f"BestF1/{method}", items))


def get_logger(
    root: str,
    name: Optional[str] = None,
    debug: bool = True,
    data: str = "swat",
    tag: str = "run",
    *,
    model: Optional[str] = None,
    run_id: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """Create a logger that logs to file and optionally to console (tqdm-friendly).

    Log file name includes time, dataset and model to distinguish runs:
      <root>/<YYYYmmdd_HHMMSS>_<DATA>_<MODEL>_<TAG>.log
    """

    os.makedirs(root, exist_ok=True)

    logger = logging.getLogger(name or f"stamp.{data}.{tag}")
    logger.setLevel(logging.DEBUG)

    # Reset handlers on each call to avoid cross-run contamination
    if getattr(logger, "_stamp_configured", False):
        return logger

    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")

    # File handler (always on)
    safe_tag = str(tag or "run").strip() or "run"
    safe_data = str(data or "data").strip()
    safe_model = str(model or name or "model").strip()
    rid = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')

    logfile = os.path.join(root, f"{rid}_{safe_data}_{safe_model}_{safe_tag}.log")
    file_handler = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Console handler (optional)
    if console:
        console_handler = TqdmConsoleHandler()
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    logger._stamp_configured = True

    # Expose log path for other tooling
    logger.logfile = logfile

    return logger


if __name__ == '__main__':
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    print(time)
    logger = get_logger('./log.txt', debug=True)
    logger.debug('this is a {} debug message'.format(1))
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')
    logger.debug('this is a debug message')
    logger.info('this is an info message')