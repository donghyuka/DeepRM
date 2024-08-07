import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import glob
from tqdm import tqdm
import argparse
import pysam
import pod5
from utils.utils import printmessage, mean_phred, seq_to_onehot

def parse_pod5(pod5_path):

    signal_list = []
    offset_list = []
    scale_list = []
    id_list = []

    try:
        with pod5.Reader(pod5_path) as reader:
            skipped = 0
            for record in reader:
                try:
                    signal_arr = record.signal
                    offset = record.calibration.offset
                    scale = record.calibration.scale
                    id = str(record.read_id)
                except:
                    skipped += 1
                    continue

                offset_list.append(offset)
                scale_list.append(scale)
                signal_list.append(signal_arr)
                id_list.append(id)

        if skipped > 0:
            printmessage(f"Skipped {skipped} faulty records in: {pod5_path}", msg_type="warning")

    except:
        ## Pod5 file is corrupted
        printmessage(f"Corrupted POD5 file: {pod5_path}", msg_type="error")
        raise ValueError

    data = pd.DataFrame({"read_id": id_list, "signal": signal_list, "offset": offset_list, "scale": scale_list})

    return data




def parse_bam(bam_path, ncpu = 16, bq_cutoff = 0, skip_unmapped = False):


    ## Extract mv tag from bam and save to separate file
    data_dict = {"mv": [], "read_id": [], "ts": [], "ns": [], "sp": [], "seq": [], "bq": [], "pt": [],
                 "ref": [], "start": [], "cigar": []}
    valid_count = 0
    missing_move = 0
    missing_bq = 0
    low_bq = 0
    missing_signal = 0
    unmapped = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm(total=input_bam.mapped + input_bam.unmapped, desc="Parsing BAM File") as pbar:
            for read in input_bam:
                pbar.update(1)
                pbar.set_postfix({"valid": valid_count, "invalid": missing_move + missing_bq + low_bq + missing_signal + unmapped})
                if skip_unmapped:
                    if read.is_unmapped:
                        unmapped += 1
                        continue

                if read.is_supplementary:
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

                data_dict["mv"].append(mv)
                data_dict["read_id"].append(read_id)
                data_dict["ts"].append(ts)
                data_dict["ns"].append(ns)
                data_dict["sp"].append(sp)
                data_dict["pt"].append(pt)
                data_dict["seq"].append(str(read.query_sequence))
                data_dict["bq"].append(bq)
                data_dict["ref"].append(read.reference_name)
                data_dict["start"].append(read.reference_start)
                data_dict["cigar"].append(read.cigarstring)


                valid_count += 1

    printmessage(f"Valid reads: {valid_count}", msg_type="info")
    printmessage(f"Low BQ: {low_bq}", msg_type="info")
    printmessage(f"Missing BQ (Secondary): {missing_bq}", msg_type="info")
    printmessage(f"Missing Signal: {missing_signal}", msg_type="info")
    printmessage(f"Missing Move: {missing_move}", msg_type="info")
    printmessage(f"Unmapped: {unmapped}", msg_type="info")

    data = pd.DataFrame(data_dict)

    data["move_sum"] = data["mv"].apply(lambda x: sum(x[1:]))
    data["seq_len"] = data["seq"].apply(lambda x: len(x))

    assert data["move_sum"].equals(data["seq_len"]), "Move sum and sequence length mismatch"

    return data



def expand_seq(move, ts, ns, seq):
    move = np.array(move)[1:]
    move = np.append(move, 1)
    move = np.where(move == 1)[0] * 6
    move = move[1:] - move[:-1]
    seq = seq_to_onehot(seq[::-1])
    seq = np.repeat(seq, move, axis=0)
    seq = np.concatenate([np.zeros((ts, 4)), seq], axis=0)
    seq = np.concatenate([seq, np.zeros((ns - seq.shape[0], 4))], axis=0)
    return seq


def plot(data):
    fig, axes = plt.subplots(figsize=(40, 0.2 * len(data)), nrows=len(data), ncols=1)
    for i, (idx, row) in tqdm(enumerate(data.iterrows()), total=len(data)):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0,20000)

        seq = row["seq"]
        seq[:,:,0] += seq[:,:,3]
        seq[:,:,1] += seq[:,:,3]
        seq = seq[:,:,:3]
        seq = seq.repeat(10, axis=0)
        ax.imshow(seq, aspect="auto")

    fig.tight_layout()
    fig.savefig("/extdata4/baeklab/Hyeonseo/m6A/plot/seq.png")
    plt.close(fig)
    return None


def main():
    pod5_df = parse_pod5("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/raw/pod5/PAU70804_pass_78353ea4_73ecad7a_777.pod5")
    yestrim_bam_df = parse_bam("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output_yestrim_sample.bam")
    notrim_bam_df = parse_bam("/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0105/ON0105/result/intermediates/dorado_output_notrim_sample.bam")

    pod5_df["signal"] = pod5_df.apply(lambda x: (x["signal"] + x["offset"]) * x["scale"], axis=1)
    yestrim_bam_df = yestrim_bam_df[yestrim_bam_df["sp"]==0]
    yestrim_bam_df = yestrim_bam_df[["read_id", "mv", "ts", "ns", "seq"]]
    yestrim_bam_df.rename({"mv": "mv_yestrim", "ts": "ts_yestrim", "ns": "ns_yestrim", "seq": "seq_yestrim"}, axis=1, inplace = True)
    notrim_bam_df = notrim_bam_df[notrim_bam_df["sp"]==0]
    notrim_bam_df = notrim_bam_df[["read_id", "mv", "ts", "ns", "seq"]]
    notrim_bam_df.rename({"mv": "mv_notrim", "ts": "ts_notrim", "ns": "ns_notrim", "seq": "seq_notrim"}, axis=1, inplace = True)

    data = pod5_df.merge(yestrim_bam_df, on="read_id", how="inner")
    data = data.merge(notrim_bam_df, on="read_id", how="inner")

    data = data.sample(100)
    data["ns_max"] = data.apply(lambda x: max(x["ns_yestrim"], x["ns_notrim"]), axis=1)
    data["seq_yestrim"] = data.apply(lambda x: expand_seq(x["mv_yestrim"], x["ts_yestrim"], x["ns_max"], x["seq_yestrim"]), axis=1)
    data["seq_notrim"] = data.apply(lambda x: expand_seq(x["mv_notrim"], x["ts_notrim"], x["ns_max"], x["seq_notrim"]), axis=1)
    data["seq_yestrim_shape"] = data["seq_yestrim"].apply(lambda x: x.shape)
    data["seq_notrim_shape"] = data["seq_notrim"].apply(lambda x: x.shape)

    data["seq"] = data.apply(lambda x: np.stack([x["seq_yestrim"], x["seq_notrim"]], axis=0), axis=1)
    print(data)

    plot(data)

    return None


if __name__ == "__main__":
    main()