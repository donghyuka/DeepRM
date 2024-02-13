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


def revcomp_RNA(seq):
    basemap = dict(zip("AUTCG", "UAAGC"))
    return "".join([basemap[i] for i in seq[::-1]])


def revcomp_DNA(seq):
    basemap = dict(zip("ATUCG", "TAAGC"))
    return "".join([basemap[i] for i in seq[::-1]])


def comp_RNA(seq):
    basemap = dict(zip("AUCG", "UAGC"))
    return "".join([basemap[i] for i in seq])


def comp_DNA(seq):
    basemap = dict(zip("ATCG", "TAGC"))
    return "".join([basemap[i] for i in seq])


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


def parse_refflat(refflat_path="/extdata4/baeklab/Hyeonseo/m6A/res/ref/GRCh38_latest_genomic.gtf.refflat.txt"):
    col_list = ["NMID", "NCID", "strand", "txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount", "exonStarts",
                "exonEnds"]
    with open(refflat_path, "r") as infile:
        refflat_df = pd.read_csv(infile, sep="\t", header=None, names=col_list)

    refflat_df["chr"] = refflat_df["NCID"].apply(lambda x: ncid_to_chr(x))
    refflat_df["NMID"] = refflat_df["NMID"].str.split(".").str[0]
    refflat_df = refflat_df[refflat_df["chr"] != "chrUnk"]
    refflat_df = refflat_df[refflat_df["cdsEnd"] >= refflat_df["cdsStart"]]
    refflat_df["chrstrand"] = refflat_df["chr"].astype(str) + refflat_df["strand"]
    refflat_df[["txStart", "txEnd", "cdsStart", "cdsEnd"]] = refflat_df[
        ["txStart", "txEnd", "cdsStart", "cdsEnd"]].astype(int)
    refflat_df["exonStarts"] = refflat_df["exonStarts"].apply(lambda x: np.array(x.split(",")[:-1]).astype(int))
    refflat_df["exonEnds"] = refflat_df["exonEnds"].apply(lambda x: np.array(x.split(",")[:-1]).astype(int))

    ## Drop duplicates
    chr_list = [f"chr{i}" for i in list(range(1, 23))] + ["X", "Y", "M"]
    refflat_df_list = []
    for chr in chr_list:
        refflat_df_chrs = refflat_df[refflat_df["chr"] == chr].copy()
        refflat_df_chrs.drop_duplicates(subset='NMID', keep="first", inplace=True, ignore_index=True)
        refflat_df_list.append(refflat_df_chrs)

    refflat_df = pd.concat(refflat_df_list, ignore_index=True)
    refflat_df.set_index("NMID", inplace=True)

    return refflat_df


def reformat_nmid(nmid):
    nmid_prefix = nmid.split("_")[0]
    nmid_suffix = nmid.split("_")[1]
    nmid_suffix = nmid_suffix.zfill(9)
    nmid = nmid_prefix + ":" + nmid_suffix
    return nmid


def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-4]


def is_drach(seq):
    seq = seq.upper()
    if re.search(r'[AGUT][AG]AC[ACUT]', seq):
        return True
    else:
        return False


def is_rrach(seq):
    seq = seq.upper()
    if re.search(r'[AG][AG]AC[ACUT]', seq):
        return True
    else:
        return False


def confidence_interval(data, level=0.95):
    assert level > 0 and level < 1, "level must be between 0 and 1"
    a = np.array(data)
    n = len(a)
    if n == 1: return np.mean(a), 0
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + level) / 2., n - 1)
    return m, h


def median(data):
    a = np.array(data)
    n = len(a)
    if n == 1: return np.mean(a), 0
    m = np.median(a)
    q75, q25 = np.percentile(a, [75, 25])
    h = (q75 - q25) / 2
    return m, h


def printmessage(*string, color=None, color_time="green", end='\n'):
    COLOR_FORE_DICT = {'red': Fore.RED, 'green': Fore.GREEN, 'yellow': Fore.YELLOW,
                       'blue': Fore.BLUE, 'magenta': Fore.MAGENTA, 'cyan': Fore.CYAN, 'white': Fore.WHITE}

    timestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestr = f"[{timestr}]"

    if color_time:
        color_time = color_time.lower()
        if color_time in COLOR_FORE_DICT:
            timestr = COLOR_FORE_DICT[color_time] + timestr + Style.RESET_ALL
        else:
            print('Warning: color_time not recognized.')

    str_out = ' '.join(str(x) for x in string)
    if color:
        color = color.lower()
        if color in COLOR_FORE_DICT:
            str_out = COLOR_FORE_DICT[color] + str_out + Style.RESET_ALL
        else:
            print('Warning: color not recognized.')

    print(timestr, str_out, end=end)

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


def mean_phred(phred):
    ## When averaging PHRED scores, note that the PHRED score is logarithmically scaled.
    return -10 * np.log10(np.mean(10 ** (-np.array(phred) / 10)))


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

