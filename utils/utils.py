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


## This file is a collection of small utility functions that are used in multiple scripts.
## The functions are not organized or documented.
## Maybe I will organize them later.
## Which almost certainly means never.
## But who knows? Maybe after the release of Half-Life 3 and Python 4.0.


def seq_to_onehot(seq:str):
    seq = seq.upper()
    seq = seq.replace('T', 'U')
    mapping = dict(zip("ACGU", range(4)))
    mapped = [mapping[i] for i in seq]
    result = np.eye(4)[mapped].astype(float)
    return result


def ncid_to_chr(ncid):
    ncid_int = int(ncid.split(".")[0][3:])
    if ncid_int <= 22:
        chr = f"chr{ncid_int}"
    elif ncid_int == 23:
        chr = "chrX"
    elif ncid_int == 24:
        chr = "chrY"
    else:
        chr = "chrUnk"
    return chr

REFFLAT_PATH = "/extdata4/baeklab/Hyeonseo/m6A/anno/agat_refflat.base0.pkl"
def parse_refflat_v2(refflat_path=REFFLAT_PATH, drop_y = False, drop_m = False, drop_unk = True, drop_ver = True, reindex = True):
    refflat_df = pd.read_pickle(refflat_path)
    refflat_df=refflat_df[refflat_df["cdsEnd"]>=refflat_df["cdsStart"]]
    refflat_df["chrstrand"]=refflat_df["chr"].astype(str)+refflat_df["strand"]
    refflat_df[["txStart","txEnd","cdsStart","cdsEnd"]]=refflat_df[["txStart","txEnd","cdsStart","cdsEnd"]].astype(int)
    refflat_df["exonStarts"]=refflat_df["exonStarts"].apply(lambda x: np.array(x.split(",")[:-1]).astype(int))
    refflat_df["exonEnds"]=refflat_df["exonEnds"].apply(lambda x: np.array(x.split(",")[:-1]).astype(int))
    return refflat_df


def reformat_transcript_id(transcript_id):
    if transcript_id.startswith("EN"):
        transcript_id = transcript_id.split(".")[0]
    return transcript_id


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


def mean_phred(phred):
    if not isinstance(phred, np.ndarray):
        phred = np.array(phred, dtype=int)
    else:
        phred = phred.astype(int)
    ## When averaging PHRED scores, note that the PHRED score is logarithmically scaled.
    return -10 * np.log10(np.mean(10 ** (-phred / 10)))


def oom_killer(program_name=None, margin=0.01):
    if program_name is None:
        program_name = os.path.basename(sys.argv[0])
    mem_total = psutil.virtual_memory().total
    mem_threshold = mem_total * margin
    if psutil.virtual_memory().available < mem_threshold:
        printmessage(f"[{program_name}] Memory usage is too high. Killing the program.")
        ## check if in subprocess
        if os.getppid() == 1:
            printmessage(f"[{program_name}] Parent process is init. Killing the program.")
            sys.exit(1)
        else:
            printmessage(f"[{program_name}] Parent process is not init. Killing the parent process.")
            os.kill(os.getppid(), 9)
            sys.exit(1)

    return None
