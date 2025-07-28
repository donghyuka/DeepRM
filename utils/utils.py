import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import psutil
from colorama import Fore, Style

## This file is a collection of small utility functions that are used in multiple scripts.

def printmessage(*string, color=None, color_time="green", end='\n', msg_type=None, error=None):
    """
    Prints a formatted message with optional color and message type.

    Args:
        *string: Message to print.
        color (str, optional): Color of the message text.
        color_time (str, optional): Color of the timestamp.
        end (str, optional): End character for the print function.
        msg_type (str, optional): Type of message (error, warning, info, success).
        error (Exception, optional): Exception to raise after printing the message.

    Returns:
        None
    """
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


def mean_phred(phred):
    """
    Calculates the mean Phred quality score.

    Args:
        phred (np.ndarray or list): Array or list of Phred quality scores.

    Returns:
        float: Mean Phred quality score.
    """
    if not isinstance(phred, np.ndarray):
        phred = np.array(phred, dtype=int)
    else:
        phred = phred.astype(int)
    return -10 * np.log10(np.mean(10 ** (-phred / 10)))


def oom_killer(program_name=None, margin=0.01):
    """
    Kills the program if memory usage exceeds a certain threshold.

    Args:
        program_name (str, optional): Name of the program.
        margin (float, optional): Memory usage threshold as a fraction of total memory.

    Returns:
        None
    """
    if program_name is None:
        program_name = os.path.basename(sys.argv[0])
    mem_total = psutil.virtual_memory().total
    mem_threshold = mem_total * margin
    if psutil.virtual_memory().available < mem_threshold:
        printmessage(f"[{program_name}] Memory usage is too high. Killing the program.")
        if os.getppid() == 1:
            printmessage(f"[{program_name}] Parent process is init. Killing the program.")
            sys.exit(1)
        else:
            printmessage(f"[{program_name}] Parent process is not init. Killing the parent process.")
            os.kill(os.getppid(), 9)
            sys.exit(1)

    return None