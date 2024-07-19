

import pod5
from utils import augmentations as aug
import glob
import numpy as np
import torch
from functools import partial
from tqdm import tqdm
import os
from utils.utils import printmessage, mean_phred
import pysam
import pandas as pd


def get_aug_dict(fraction):

    ## MOVING AVERAGE MAGNITUDE WARP
    movmag = partial(aug.moving_magnitude_warp, fraction=fraction, min_sigma=0.1, max_sigma=0.2, n_knots = 10, window_size=5)
    ## WINDOWED TIME WARP
    winwarp = partial(aug.window_warp, fraction=fraction, min_window_ratio = 0.05, max_window_ratio = 0.10,
                      min_window_count = 5, max_window_count = 10, sigma = 0.2)
    ## TIME WARP
    timewarp = partial(aug.time_warp, fraction=fraction, min_sigma=0.1, max_sigma=0.2, n_knots = 30)
    ## GAUSSIAN JITTER
    jitter = partial(aug.jitter, fraction=fraction, min_sigma=0.2, max_sigma=0.5)
    ## SPIKE NOISE
    spike = partial(aug.jitter, fraction=fraction, min_sigma=0.2, max_sigma=0.5, dropout = 0.9)
    ## STEP NOISE
    step = partial(aug.step, fraction=fraction, min_sigma=0.2, max_sigma=0.5, dropout = 0.9)
    ## SLOPE NOISE
    slope = partial(aug.slope, fraction=fraction, magnitude = 0.3,)
    ## DRIFT NOISE
    drift = partial(aug.drift, fraction=fraction, min_sigma=0.2, max_sigma=0.5, n_knots = 10)

    aug_dict = {"movmag": movmag, "winwarp": winwarp, "timewarp": timewarp, "jitter": jitter, "spike": spike, "step": step, "slope": slope, "drift": drift}
    return aug_dict

def augment_signal(pod5_path):
    signal_list = []
    read_list = []
    with pod5.Reader(pod5_path) as reader:
        for record in tqdm(reader, total=reader.num_reads, desc="Reading pod5"):
            signal = torch.tensor(record.signal).float()
            if signal.min() <= 0:
                continue
            signal_list.append(signal)
            read_list.append(record.to_read())

    signal = torch.nn.utils.rnn.pad_sequence(signal_list, batch_first=True, padding_value=0)

    print(signal)
    print(signal.shape)

    aug_dict = get_aug_dict(1.0)
    sig_dict = {"original": signal}
    signal_split = torch.split(signal, 1000, dim=1)

    for key, aug_func in tqdm(aug_dict.items(), total=len(aug_dict), desc="Augmenting signals"):
        signal_aug = torch.cat([aug_func(split, pad_mask = (split!=0)) for split in signal_split], dim=1).round().int()
        print(key, signal_aug)
        sig_dict[key] = signal_aug

    return sig_dict, read_list

def write_pod5(sig_dict, wdir, read_list):
    for key, signal in sig_dict.items():
        if os.path.exists(f"{wdir}/{key}.pod5"):
            os.remove(f"{wdir}/{key}.pod5")
        with pod5.Writer(f"{wdir}/{key}.pod5") as writer:
            for idx, record in tqdm(enumerate(read_list), desc=f"Writing {key} pod5", total=len(read_list)):
                read_signal = signal[idx]
                read_signal = read_signal[read_signal.nonzero(as_tuple=True)].numpy()
                record.signal = read_signal
                writer.add_read(record)

    return None


def run_basecall(sig_dict, wdir):
    pod_list = glob.glob(f"{wdir}/*.pod5")
    dorado = "/extdata2/baeklab/Hyeonseo/bin/dorado-0.7.2"
    dorado_model = "/extdata2/baeklab/Hyeonseo/bin/dorado-0.7.2/model/rna004_130bps_sup@v5.0.0"
    cmd = "{dorado}/bin/dorado basecaller --chunksize 12000 -x cuda:all --min-qscore 0 --emit-moves --estimate-poly-a {dorado_model} {pod5_file} > {bam_file}"

    for pod5_file in pod_list:
        name = os.path.basename(pod5_file).split(".")[0]
        if name not in sig_dict:
            continue
        printmessage(f"Running Dorado on {pod5_file}")
        bam_file = pod5_file.replace(".pod5", ".bam")
        os.system(cmd.format(dorado=dorado, dorado_model=dorado_model, pod5_file=pod5_file, bam_file=bam_file))
        os.system(f"samtools sort -@ 10 -o {bam_file} {bam_file}")
        os.system(f"samtools index {bam_file}")

    return None

def merge_bams(wdir):
    bam_list = glob.glob(f"{wdir}/*.bam")
    df_list = []
    name_list = []
    for bam_file in bam_list:
        name = os.path.basename(bam_file).split(".")[0]
        name_list.append(name)
        bam_dict = {}
        with pysam.AlignmentFile(bam_file, "rb", check_sq=False) as bam:
            for read in bam:
                read_id = read.query_name
                seq = read.query_sequence
                phred = mean_phred(read.query_qualities)
                bam_dict[read_id] = [seq, phred]
        df = pd.DataFrame.from_dict(bam_dict, orient="index", columns=[f"{name}_seq", f"{name}_bq"])
        df_list.append(df)
        print(df[f"{name}_bq"].describe())
    df = pd.concat(df_list, axis=1)
    print(df)
    df.to_csv(f"{wdir}/merged.tsv", sep="\t")

    ## Plot KDE of base quality
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(20, 10))
    for name in name_list:
        sns.kdeplot(df[f"{name}_bq"], ax=ax, label=name)
    ax.set_xlabel("Base Quality")
    ax.set_ylabel("Density")
    ax.set_title("Base Quality Density")
    ax.legend()
    plt.savefig(f"{wdir}/merged_bq_kde.png")
    plt.close()
    return None




def main():
    pod5_path = "/extdata2/baeklab/Hyeonseo/m6A/analyses/original.pod5"
    wdir = "/extdata2/baeklab/Hyeonseo/m6A/analyses/aug_dorado"
    sig_dict, read_list = augment_signal(pod5_path)
    write_pod5(sig_dict, wdir, read_list)
    run_basecall(sig_dict, wdir)
    merge_bams(wdir)
    return None


if __name__ == "__main__":
    main()