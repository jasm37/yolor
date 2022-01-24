
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

LOG_LEVEL = logging.DEBUG

LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def colorstr(*args: Any) -> str:
    """Make color string🌈.
    Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code,
    i.e.  colorstr('blue', 'hello world')

    Color codes:
        "black",
        "red",
        "green",
        "yellow",
        "blue", (Default)
        "magenta",
        "cyan",
        "white",
        "bright_black",
        "bright_red",
        "bright_green",
        "bright_yellow",
        "bright_blue",
        "bright_magenta",
        "bright_cyan",
        "bright_white",

    Text format:
        "bold",  (Default)
        "underline",
    :param args: string with text format
                  Ex) colorstr("red", "bold" "Hello world")
                      will print red and bold text of "Hello world"
    :return: text with color🌈
    """
    *args, string = (
        args if len(args) > 1 else ("blue", "bold", args[0])  # type: ignore
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }

    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


# Logger config
COLOR_FORMATTER = logging.Formatter(
    "[%(asctime)s]"
    + colorstr("yellow", "bold", "[%(levelname)s]")
    # + colorstr("green", "bold", "[%(name)s]")
    + colorstr("cyan", "bold", "[%(filename)s:%(lineno)d]")
    + colorstr("blue", "bold", "(%(funcName)s)")
    + " %(message)s"
)
GREY_FORMATTER = logging.Formatter(
    "[%(asctime)s]"
    + "[%(levelname)s]"
    + "[%(filename)s:%(lineno)d]"
    + "(%(funcName)s)"
    + " %(message)s"
)


def get_logger(
        name: str = "Logger", log_level: Optional[Union[str, int]] = None, main_proc_only: bool = True
) -> logging.Logger:
    """Get logger with formatter
    :param name: logger name
    :param log_level: logging level if None is given, constants.LOG_LEVEL will be used
    :param main_proc_only: log only rank in [-1, 0]
    :return: logger with string formatter
    """
    _logger = logging.getLogger(name)
    _logger.setLevel(LOG_LEVEL)

    if main_proc_only and RANK not in [-1, 0]:
        _logger.disabled = True
    elif not _logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(COLOR_FORMATTER)
        if log_level is None:
            log_level = LOG_LEVEL
        elif isinstance(log_level, str):
            log_level = log_level.upper()
        elif not isinstance(log_level, int):
            raise "Log level must be either a string or an integer"
        ch.setLevel(log_level)
        _logger.addHandler(ch)
        _logger.propagate = False
    return _logger


logger = get_logger(log_level="debug")


def save_logs(filepath: Union[str, Path]):
    """Sets a logger handler to store logs in the given path
    :param filepath: log file name
    :return instantiated logger
    """
    file_handler = logging.FileHandler(filename=filepath)
    file_handler.setFormatter(GREY_FORMATTER)
    logger.addHandler(file_handler)
    return logger
