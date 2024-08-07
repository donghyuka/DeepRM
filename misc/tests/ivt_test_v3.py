import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import glob
import tqdm
from utils.utils import printmessage, mean_phred
import pysam
import pickle
import gc
import os


def standardise_trim_segment_signal(signal,move,sp,ts,ns,offset,scale):
    signal = signal[sp:]
    signal_len = len(signal)
    if ns == 0:
        ns = signal_len
    signal = signal[ts:ns]
    if len(signal) == 0:
        return None
    signal = np.flip(signal, axis=0)
    signal = (signal + offset) * scale

    move = np.array(move, dtype=int)
    stride = move[0]
    move = move[1:]
    move_idx = len(signal) - (np.nonzero(move)[0][1:] * stride)
    move_idx = np.flip(move_idx, axis=0)
    signal = np.array_split(signal, move_idx)
    if len(signal) < 5:
        return None
    signal = np.array([np.mean(x) for x in signal])
    return signal

#
# def extract_kmer_signal(path, nmid="NM_001101.5"):
#
#     s_paths = f"{path}/signal_raw/*.pkl"
#     s_paths = glob.glob(s_paths)
#     b_paths = [x.replace("signal_raw", "move_df_split") for x in s_paths]
#
#     df_list = []
#
#     for s_path, b_path in tqdm(zip(s_paths, b_paths), desc="Extracting kmer signal", total=len(s_paths)):
#         try:
#             data2 = pd.read_pickle(b_path)
#             data2 = data2[data2["ref"] == nmid]
#             data = pd.read_pickle(s_path)
#         except:
#             continue
#         data = data.merge(data2, on="read_id", how="inner")
#         df_list.append(data)
#
#     return df



def extract_move(bam_path, ncpu, bq_cutoff, intermediate_path, output_name="debug_actb", skip_unmapped = True):

    signal_index_path = f"{intermediate_path}/signal_index.pkl"
    with open(signal_index_path, "rb") as infile:
        index_dict = pickle.load(infile)

    signal_path_dict = {}
    for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Read-to-File Index"):
        for read_id in id_list:
            signal_path_dict[read_id] = signal_path.split('/')[-1]
    signal_path_arr = [x.split('/')[-1] for x in list(index_dict.keys())]

    ## Extract mv tag from bam and save to separate file
    data_dict = {x: {"mv": [], "read_id": [], "ts": [], "ns": [], "sp": [], "seq": [], "bq": [], "pt": [],
                     "ref": [], "start": [], "cigar": []} for x in signal_path_arr}
    valid_count = 0
    missing_move = 0
    missing_bq = 0
    low_bq = 0
    missing_signal = 0
    unmapped = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped + input_bam.unmapped, desc="Parsing BAM File") as pbar:
            for read in input_bam:
                pbar.update(1)
                pbar.set_postfix({"valid": valid_count, "invalid": missing_move + missing_bq + low_bq + missing_signal + unmapped})
                if skip_unmapped:
                    if read.is_unmapped:
                        unmapped += 1
                        continue

                if read.has_tag("pi"):
                    read_id = str(read.get_tag("pi"))
                else:
                    read_id = str(read.query_name)

                try:
                    bq = np.array(read.query_qualities, dtype=int)
                    if mean_phred(bq) < bq_cutoff:
                        low_bq += 1
                        continue
                except:
                    missing_bq += 1
                    continue

                try:
                    signal_path = signal_path_dict[read_id]
                    data = data_dict[signal_path]
                except:
                    missing_signal += 1
                    continue

                if read.has_tag("mv"):
                    mv = read.get_tag("mv")
                else:
                    missing_move += 1
                    continue

                if read.has_tag("ts"):
                    ts = read.get_tag("ts")
                else:
                    ts = 0

                if read.has_tag("ns"):
                    ns = read.get_tag("ns")
                else:
                    ns = 0

                if read.has_tag("sp"):
                    sp = read.get_tag("sp")
                else:
                    sp = 0

                if read.has_tag("pt"):
                    pt = read.get_tag("pt")
                else:
                    pt = 0

                data["mv"].append(mv)
                data["read_id"].append(read_id)
                data["ts"].append(ts)
                data["ns"].append(ns)
                data["sp"].append(sp)
                data["pt"].append(pt)
                data["seq"].append(str(read.query_sequence))
                data["bq"].append(bq)
                data["ref"].append(read.reference_name)
                data["start"].append(read.reference_start)
                data["cigar"].append(read.cigarstring)

                valid_count += 1

    printmessage(f"Valid reads: {valid_count}", msg_type="info")
    printmessage(f"Low BQ: {low_bq}", msg_type="info")
    printmessage(f"Missing BQ (Secondary): {missing_bq}", msg_type="info")
    printmessage(f"Missing Signal: {missing_signal}", msg_type="info")
    printmessage(f"Missing Move: {missing_move}", msg_type="info")
    printmessage(f"Unmapped: {unmapped}", msg_type="info")

    os.makedirs(f"{intermediate_path}/{output_name}", exist_ok=True)

    for signal_path, data in tqdm.tqdm(data_dict.items(), total=len(data_dict), desc="Saving Move Data"):
        move_df = pd.DataFrame.from_dict(data, orient="columns")
        df_len = len(move_df)
        if df_len > 0:
            move_df.to_pickle(f"{intermediate_path}/{output_name}/{signal_path}")
        del move_df

    del data_dict

    gc.collect()
    return None


def main():
    # cell_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block_070/intermediates"
    # ivt_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/block/intermediates"
    # cell_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/dorado-070-aligned/intermediates/dorado_output.actb.bam"
    # ivt_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output.actb.bam"
    #
    # extract_move(cell_bam, 120, 7, cell_path)
    # extract_move(ivt_bam, 120, 7, ivt_path)
    #
    # ivt_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0104/result/block/intermediates"
    # ivt_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0104/result/intermediates/dorado_output.actb.bam"
    # extract_move(ivt_bam, 120, 7, ivt_path)

    # ivt_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA
    # /ON0105/ON0105/result/block/intermediates"
    # ivt_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output_notrim_noalign.bam"
    # extract_move(ivt_bam, 120, 7, ivt_path, output_name="debug_notrim_noalign", skip_unmapped=False)

    # ivt_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/block/intermediates"
    # ivt_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output_manualtrim.bam"
    # extract_move(ivt_bam, 120, 7, ivt_path, output_name="debug_manualtrim", skip_unmapped=True)

    ivt_path = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/block/intermediates"

    # ivt_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output_example_noalign_notrim.bam"
    # extract_move(ivt_bam, 120, 7, ivt_path, output_name="example_noalign_notrim", skip_unmapped=False)
    #
    # ivt_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output_example_noalign_yestrim.bam"
    # extract_move(ivt_bam, 120, 7, ivt_path, output_name="example_noalign_yestrim", skip_unmapped=False)
    #
    # ivt_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output_example_yesalign_notrim.bam"
    # extract_move(ivt_bam, 120, 7, ivt_path, output_name="example_yesalign_notrim", skip_unmapped=False)
    #
    # ivt_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output_example_yesalign_yestrim.bam"
    # extract_move(ivt_bam, 120, 7, ivt_path, output_name="example_yesalign_yestrim", skip_unmapped=False)

    ivt_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output_yestrim_sample.bam"
    extract_move(ivt_bam, 120, 7, ivt_path, output_name="dorado_output_yestrim_sample", skip_unmapped=False)

    ivt_bam = "/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output_notrim_sample.bam"
    extract_move(ivt_bam, 120, 7, ivt_path, output_name="dorado_output_notrim_sample", skip_unmapped=False)

    return None

if __name__ == "__main__":
    main()