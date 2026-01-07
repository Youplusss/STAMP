import os
import logging
from datetime import datetime
from typing import Optional


class TqdmConsoleHandler(logging.StreamHandler):
    """Console handler compatible with tqdm progress bars."""

    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            super().emit(record)


def log_hparams(logger: logging.Logger, args, keys: Optional[list[str]] = None) -> None:
    """Log a compact hyperparameter summary.

    We keep it readable and stable across runs; by default we log a curated subset.
    """
    try:
        d = vars(args)
    except Exception:
        d = {}

    keys = keys or [
        'data', 'model', 'pred_model', 'recon_model',
        'pred_lr_init', 'ae_lr_init',
        'batch_size', 'epochs',
        'is_down_sample', 'down_len',
        'is_mas', 'mamba_use_mas',
        'use_adv', 'adv_freeze_other',
        'grad_clip', 'max_grad_norm',
        'seed', 'gpu_id',
    ]

    lines = []
    for k in keys:
        if k in d:
            lines.append(f"- {k}: {d.get(k)}")

    logger.info("[Hyperparameters]\n" + "\n".join(lines))


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