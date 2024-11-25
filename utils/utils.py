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

REFFLAT_PATH = "/extdata4/baeklab/Hyeonseo/m6A/res/ref/GRCh38_latest_genomic.gtf.refflat.txt"
def parse_refflat(refflat_path=REFFLAT_PATH, drop_y = False, drop_m = True, drop_unk = True, drop_ver = True, reindex = True):
    col_list=["NMID","NCID","strand","txStart","txEnd","cdsStart","cdsEnd","exonCount","exonStarts","exonEnds"]
    with open(refflat_path,"r") as infile:
        refflat_df=pd.read_csv(infile,sep="\t",header=None,names=col_list)

    refflat_df["chr"]=refflat_df["NCID"].apply(lambda x: ncid_to_chr(x))
    if drop_ver:
        refflat_df["NMID"]=refflat_df["NMID"].str.split(".").str[0]
    if drop_unk:
        refflat_df = refflat_df[refflat_df["chr"]!="chrUnk"]
    refflat_df=refflat_df[refflat_df["cdsEnd"]>=refflat_df["cdsStart"]]
    refflat_df["chrstrand"]=refflat_df["chr"].astype(str)+refflat_df["strand"]
    refflat_df[["txStart","txEnd","cdsStart","cdsEnd"]]=refflat_df[["txStart","txEnd","cdsStart","cdsEnd"]].astype(int)
    refflat_df["exonStarts"]=refflat_df["exonStarts"].apply(lambda x: np.array(x.split(",")[:-1]).astype(int))
    refflat_df["exonEnds"]=refflat_df["exonEnds"].apply(lambda x: np.array(x.split(",")[:-1]).astype(int))


    ## Drop duplicates
    add_list = ["X"]
    if not drop_y:
        add_list.append("Y")
    if not drop_m:
        add_list.append("M")
    chr_list = [f"chr{i}" for i in list(range(1,23))+add_list]
    refflat_df_list=[]
    for chr in chr_list:
        refflat_df_chrs = refflat_df[refflat_df["chr"]==chr].copy()
        refflat_df_chrs.drop_duplicates(subset='NMID',keep="first",inplace=True,ignore_index=True)
        refflat_df_list.append(refflat_df_chrs)

    refflat_df = pd.concat(refflat_df_list,ignore_index=True)

    if reindex:
        refflat_df.set_index("NMID",inplace=True)

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
        print(timestr, str_type, str_out, end=end)

    else:
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

def max_f1_score(y_true, y_pred, return_threshold=False):
    from sklearn.metrics import f1_score
    f1_scores = []
    for threshold in np.arange(0, 1, 0.01):
        y_pred_ = y_pred > threshold
        f1_scores.append(f1_score(y_true, y_pred_))
    argmax_f1 = np.argmax(f1_scores)
    max_f1 = f1_scores[argmax_f1]
    threshold = argmax_f1 / 100
    if return_threshold:
        return max_f1, threshold
    else:
        return max_f1


def transcript_to_chromosomal_coordinate(coord,chrom,exon_starts,exon_ends,strand,exon_cumsum):

    try:

        if strand == "+":
            exon_index = np.searchsorted(exon_cumsum, coord, side="right") - 1
            chrom_coord = exon_starts[exon_index] + (coord - exon_cumsum[exon_index])
        elif strand == "-":
            exon_index = np.searchsorted(exon_cumsum, coord, side="right") - 1
            chrom_coord = exon_ends[-exon_index-1] - (coord - exon_cumsum[exon_index]) - 1
        else:
            raise ValueError("Invalid strand")

        genome_id = f"{chrom}:{strand}:{chrom_coord}"

    except IndexError:
        ## This is Poly(A) tail
        return None

    return genome_id


def chromosomal_to_transcript_coordinate(nmid, chr_coordinate, chrstrand, strand, refflat_df):
    ## Purpose: convert chromosome coordinate to mrna coordinate, based on exon starts and ends
    ## Expects and returns zero-based coordinates

    ## Get the exon starts and ends
    refflat_df_chrs = refflat_df[refflat_df["chrstrand"]==chrstrand]
    exon_starts = refflat_df_chrs.loc[nmid,"exonStarts"]
    exon_ends = refflat_df_chrs.loc[nmid,"exonEnds"]


    exons=np.stack((exon_starts,exon_ends),axis=1)
    exon_cumsum=np.concatenate(([0],np.cumsum(exons[:,1]-exons[:,0])))
    try:
        exon_index = np.searchsorted(exons[:,0],chr_coordinate,side="right")-1
    except:
        print(nmid,chr_coordinate,chrstrand,strand)
        raise ValueError
    mrna_coordinate=exon_cumsum[exon_index]+(chr_coordinate-exons[exon_index,0])
    mrna_length=exon_cumsum[-1]

    if strand=="-":
        mrna_coordinate=mrna_length-mrna_coordinate-1

    return mrna_coordinate


def check_in_exon(pos,exon_starts,exon_ends):
    exon_zip = zip(exon_starts,exon_ends)
    in_exon = [(pos >= exon_start) & (pos < exon_end) for exon_start,exon_end in exon_zip]
    in_exon = np.array(np.any(in_exon))

    return in_exon


def match_site_to_gene(chrstrand, pos, refflat_df):
    refflat_df_chrstrand = refflat_df[refflat_df["chrstrand"]==chrstrand]
    refflat_df_gene = refflat_df_chrstrand[(refflat_df_chrstrand["txStart"]<=pos) & (refflat_df_chrstrand["txEnd"]>pos)]

    if len(refflat_df_gene)==0:
        return []

    ## Check if the coordinate is in the exon.
    refflat_df_gene["in_exon"]=refflat_df_gene.apply(lambda x: check_in_exon(pos,x["exonStarts"],x["exonEnds"]),axis=1)
    refflat_df_gene = refflat_df_gene[refflat_df_gene["in_exon"]==True]

    return refflat_df_gene.index.to_numpy()


def chromosomal_to_transcript_df(m6a_df, refflat_df, pos_col, strand_col, m6a_df_list):

    m6a_df["NMID"]=m6a_df.apply(lambda x: match_site_to_gene(x["chrstrand"],x[pos_col],refflat_df),axis=1)

    ## Filter out sites that do not match to a gene
    m6a_df = m6a_df[m6a_df["NMID"].apply(lambda x: len(x)>0)]

    ## Explode the NMID column
    m6a_df = m6a_df.explode("NMID")

    ## Second, convert chromosome coordinate to transcript coordinate
    m6a_df["transcript_coordinate"]=m6a_df.apply(lambda x: chromosomal_to_transcript_coordinate(
        x["NMID"],x[pos_col], x["chrstrand"], x[strand_col],refflat_df),axis=1)

    m6a_df_list.append(m6a_df)

    return None


def transcript_to_chromosomal_func(label_id, refflat_df):
    nmid = label_id.split(":")[0]
    pos = int(label_id.split(":")[1])
    refflat_row = refflat_df.loc[nmid]
    chrom, strand = refflat_row["chr"], refflat_row["strand"]
    exon_starts, exon_ends = refflat_row["exonStarts"], refflat_row["exonEnds"]
    exon_cumsum = np.cumsum(np.array(exon_ends) - np.array(exon_starts))
    genome_id = transcript_to_chromosomal_coordinate(pos, chrom,exon_starts,exon_ends,strand,exon_cumsum)
    return genome_id

def transcript_to_chromosomal(transcript_id_array, refflat_df, threads=120):
    def convert_worker(transcript_id_array, refflat_df, return_list):
        local_collect = []
        for transcript_id in transcript_id_array:
            genome_id = [transcript_to_chromosomal_func(label_id, refflat_df) for label_id in transcript_id]
            local_collect.append(genome_id)
        return_list.append(local_collect)
        return None

    transcript_id_array_split = np.array_split(transcript_id_array, threads)
    man = mp.Manager()
    return_list = man.list()
    proc_list = []
    for transcript_id_array in transcript_id_array_split:
        proc = mp.Process(target=convert_worker, args=(transcript_id_array, refflat_df, return_list))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    genome_id_list = np.concatenate(return_list)
    return genome_id_list
