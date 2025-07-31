"""
Memory management utilities for DeepRM.
"""

import logging
import os
import signal
import threading
import time

import psutil

from deeprm.utils.logging import get_logger


def _check_once(limit_gb: float, logger: logging.Logger) -> bool:
    """
    Check if the resident set size (RSS) exceeds the given limit in GiB.

    Args:
        limit_gb (float): Memory limit in GiB.
        logger (logging.Logger): Logger to log messages.

    Returns:
        bool: True if RSS exceeds limit, False otherwise.
    """
    rss_gb = psutil.Process().memory_info().rss / (1024**3)
    if rss_gb > limit_gb:
        logger.error(f"RSS {rss_gb:.2f} GiB > limit {limit_gb:.2f} GiB – exiting.")
        return True
    return False


def start_mem_watchdog(limit_gb: float = None, interval_s: int = 10) -> threading.Thread:
    """
    Start a daemon thread that exits the *current process* if RSS exceeds limit.

    Args:
        limit_gb (float): Memory limit in GiB. Defaults to 95% of total RAM. (optional)
        interval_s (int): Check interval in seconds. Defaults to 10. (optional)

    Returns:
        threading.Thread: The watchdog thread.
    """
    if limit_gb is None:
        limit_gb = psutil.virtual_memory().total / (1024**3) * 0.95  # Default to 95% of total RAM

    def _run():
        log = get_logger(__name__)
        while True:
            if _check_once(limit_gb, log):
                os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(interval_s)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
