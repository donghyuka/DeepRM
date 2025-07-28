import logging
import sys
from colorama import init as _colorama_init, Fore, Style

_colorama_init(autoreset=True)

_LEVEL_COLOR = {
    logging.DEBUG:    Fore.BLUE,
    logging.INFO:     Fore.CYAN,
    logging.WARNING:  Fore.YELLOW,
    logging.ERROR:    Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}

class ColorFormatter(logging.Formatter):
    def __init__(self, datefmt: str = "%Y-%m-%d %H:%M:%S"):
        super().__init__()
        self.datefmt = datefmt

    def format(self, record: logging.LogRecord) -> str:
        # time prefix
        asctime = self.formatTime(record, self.datefmt)
        prefix = f"[{asctime}]"

        # main message (preserve logging’s lazy %-formatting)
        message = record.getMessage()

        # indent multiline messages to align with the prefix
        pad = " " * (len(prefix) + 1)
        if "\n" in message:
            message = message.replace("\n", "\n" + pad)

        color = _LEVEL_COLOR.get(record.levelno, "")
        reset = Style.RESET_ALL if color else ""

        # optional level tag (kept subtle; comment in if you want it)
        # level_tag = f"[{record.levelname.lower()}] "
        # return f"{prefix} {color}{level_tag}{message}{reset}"

        return f"{prefix} {color}{message}{reset}"

def get_logger(name: str = "deeprm", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with colored console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        # Don’t propagate to root to avoid duplicate lines in some environments
        logger.propagate = False
    return logger
