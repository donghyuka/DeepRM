import os
import re
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import psutil
from colorama import Fore, Style
from scipy import stats
import multiprocessing as mp


def printmessage(*string, color=None, color_time="green", end='\n', msg_type=None, error=None):
    COLOR_FORE_DICT = {'red': Fore.RED, 'green': Fore.GREEN, 'yellow': Fore.YELLOW,
                       'blue': Fore.BLUE, 'magenta': Fore.MAGENTA, 'cyan': Fore.CYAN, 'white': Fore.WHITE}

    timestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestr = f"[{timestr}]"

    if color_time:
        color_time = color_time.lower()
        if color_time in COLOR_FORE_DICT:
            timestr = COLOR_FORE_DICT[color_time] + timestr + Style.RESET_ALL
        else:
            print(timestr, '[Printmessage warning] Argument color_time not recognized.')

    str_out = ' '.join(str(x) for x in string)

    if color:
        color = color.lower()
        if color in COLOR_FORE_DICT:
            str_out = COLOR_FORE_DICT[color] + str_out + Style.RESET_ALL
        else:
            print(timestr, '[Printmessage warning] Argument color not recognized.')

    if msg_type is not None:
        if msg_type == 'error':
            str_type = COLOR_FORE_DICT['red'] + '[error]' + Style.RESET_ALL
        elif msg_type == 'warning':
            str_type = COLOR_FORE_DICT['yellow'] + '[warning]' + Style.RESET_ALL
        elif msg_type == 'info':
            str_type = COLOR_FORE_DICT['cyan'] + '[info]' + Style.RESET_ALL
        elif msg_type == 'success':
            str_type = COLOR_FORE_DICT['green'] + '[success]' + Style.RESET_ALL
        else:
            str_type = COLOR_FORE_DICT['white'] + f'[{msg_type}]' + Style.RESET_ALL
        timestr = f"{timestr} {str_type}"

    len_timestr = len(timestr)
    str_out = str_out.replace('\n', f'\n{" " * len_timestr}')
    print(timestr, str_out, end=end)

    if error is not None:
        if not (isinstance(error, Exception) or issubclass(error, Exception)):
            raise ValueError(f"[Printmessage error] Error argument must be an exception, not {type(error)}")
        else:
            raise error

    return None


def print_in_box(msg, indent=1, width=None, title=None, color=None):
    ## Print message-box with optional title.
    ## Based on: https://stackoverflow.com/questions/39969064/how-to-print-a-message-box-in-python

    COLOR_FORE_DICT = {'red': Fore.RED, 'green': Fore.GREEN, 'yellow': Fore.YELLOW,
                       'blue': Fore.BLUE, 'magenta': Fore.MAGENTA, 'cyan': Fore.CYAN, 'white': Fore.WHITE}
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line.strip():<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border

    if color:
        color = color.lower()
        if color in COLOR_FORE_DICT:
            fore = COLOR_FORE_DICT[color]
            box = fore + box + Style.RESET_ALL
        else:
            print(f'Warning: color {color} not supported')

    print(box)
    return None


class Timeit(object):
    def __init__(self, name, verbose=True):
        self.name = name
        self.start = time.time()
        self.end = None
        self.verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        if self.verbose:
            print(f"{self.name} took {self.end - self.start} seconds")

