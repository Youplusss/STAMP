import os
import logging
from datetime import datetime


def get_logger(root, name=None, debug=True, data="swat", tag: str = "run"):
    """Create a logger that always logs to file and console.

    Parameters
    ----------
    root:
        Log directory.
    name:
        Logger name.
    debug:
        If True, console shows DEBUG; else INFO.
    data:
        Dataset name used in filename.
    tag:
        A suffix to separate logs, e.g. 'train' / 'test'.

    Previous behavior only wrote to file when debug=False, which explains why
    many *_run.log files stayed empty.

    This version:
    - Always writes DEBUG-level logs to <root>/<data>_<tag>.log.
    - Writes DEBUG/INFO to console depending on debug.
    - Avoids accumulating duplicate handlers across multiple calls.
    """

    os.makedirs(root, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers when get_logger is called multiple times
    if getattr(logger, "_stamp_configured", False):
        return logger

    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler (always on)
    safe_tag = str(tag or "run").strip() or "run"
    logfile = os.path.join(root, f"{data}_{safe_tag}.log")
    print('Creat Log File in: ', logfile)
    file_handler = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False
    logger._stamp_configured = True
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