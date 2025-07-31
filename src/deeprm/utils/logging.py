"""
Logging utilities for DeepRM.
"""

import logging
import sys

from colorama import Fore, Style
from colorama import init as _colorama_init

_colorama_init(autoreset=True)

_LEVEL_COLOR = {
    logging.DEBUG: Fore.BLUE,
    logging.INFO: Fore.CYAN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}


class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter that adds colored output and time prefix.
    It formats log messages with a timestamp and applies colors based on the log level.
    It also indents multiline messages to align with the timestamp prefix.

    Args:
        datefmt (str): Format string for the timestamp. Defaults to "%Y-%m-%d %H:%M:%S".

    Attributes:
        datefmt (str): Format string for the timestamp.
    """

    def __init__(self, datefmt: str = "%Y-%m-%d %H:%M:%S"):
        super().__init__()
        self.datefmt = datefmt

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with a timestamp and colored output.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message with timestamp and color.
        """
        # time prefix
        asctime = self.formatTime(record, self.datefmt)
        prefix = f"[{asctime}] {record.levelname}"

        # main message (preserve logging’s lazy %-formatting)
        message = record.getMessage()

        # indent multiline messages to align with the prefix
        pad = " " * (len(prefix) + 1)
        if "\n" in message:
            message = message.replace("\n", "\n" + pad)

        color = _LEVEL_COLOR.get(record.levelno, "")
        reset = Style.RESET_ALL if color else ""

        return f"{prefix} {color}{message}{reset}"


def get_logger(name: str = "deeprm", level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger with colored console output.

    Args:
        name (str): Name of the logger. Defaults to "deeprm".
        level (int): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
