import numpy as np
import pickle
from Bio import SeqIO
from tqdm import tqdm
import RNA
import argparse
import multiprocessing as mp
import os
from utils.utils import printmessage
import pandas as pd
import shutil
import seaborn as sns
import matplotlib.pyplot as plt


def get_keys_from_dict(path):
    printmessage("Loading Coordinates")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.array(list(data.keys()))


def get_keys_from_df(path, colname = "label_id"):
    printmessage("Loading Coordinates")
    data = pd.read_pickle(path)
    return data[colname].unique()

def get_keys_from_tsv(path):
    printmessage("Loading Coordinates")
    data = pd.read_csv(path, sep = "\t", header = None)
    return data[0].values

def group_coordinates(keys):
    printmessage("Grouping Coordinates")
    group = {}
    for key in tqdm(keys, desc = "Grouping"):
        transcript_id = key.split(":")[0]
        if transcript_id not in group:
            group[transcript_id] = []
        group[transcript_id].append(key)
    return group


def split_fasta_and_coordinate(coordinate_dict, fasta_dict, threads = 10):
    printmessage("Splitting FASTA and Coordinates")
    sample_count_dict = {key: len(value) for key, value in coordinate_dict.items()}
    sorted_keys = sorted(sample_count_dict.items(), key = lambda x: x[1], reverse = True)
    split_keys = np.array_split([x[0] for x in sorted_keys], threads)
    split_fasta_dict = {}
    split_coordinate_dict = {}
    for i, keys in tqdm(enumerate(split_keys), total = threads, desc = "Splitting"):
        split_fasta_dict[i] = {key: fasta_dict[key] for key in keys}
        split_coordinate_dict[i] = {key: coordinate_dict[key] for key in keys}
    return split_fasta_dict, split_coordinate_dict


def get_fasta_as_dict(path):
    printmessage("Loading FASTA")
    with open(path, "r") as f:
        data = SeqIO.to_dict(SeqIO.parse(f, "fasta"))
    return data


def get_mfe(pid, output, coordinate_dict, fasta_dict, pad):
    mfe_dict = {}
    for transcript_id, coordinates in tqdm(coordinate_dict.items(), total = len(coordinate_dict), desc = "Calculating MFE"):
        seq = str(fasta_dict[transcript_id].seq)
        seq_len = len(seq)
        for coordinate in coordinates:
            centre = coordinate.split(":")[1]
            centre = int(centre)
            if centre - pad < 0 or centre + pad >= seq_len:
                mfe_dict[coordinate] = None
            else:
                mfe = RNA.fold(seq[centre-pad:centre+pad+1])
                mfe_dict[coordinate] = mfe[1]

    with open(f"{output}/temp/mfe_{pid}.pkl", "wb") as f:
        pickle.dump(mfe_dict, f)

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinate", "-c", type = str, required = True)
    parser.add_argument("--fasta", "-f", type = str, required = True)
    parser.add_argument("--output", "-o", type = str, required = True)
    parser.add_argument("--threads", "-t", type = int, default = 120)
    parser.add_argument("--pad", "-p", type = int, default = 10)
    return parser.parse_args()


def main():
    printmessage("Start")
    args = parse_args()
    coordinate_keys = get_keys_from_df(args.coordinate)
    coordinate_group = group_coordinates(coordinate_keys)
    fasta_dict = get_fasta_as_dict(args.fasta)
    split_fasta_dict, split_coordinate_dict = split_fasta_and_coordinate(coordinate_group, fasta_dict, threads = args.threads)
    proc_list = []
    os.makedirs(args.output, exist_ok = True)
    os.makedirs(f"{args.output}/temp", exist_ok = True)
    for i in range(args.threads):
        proc = mp.Process(target = get_mfe, args = (i, args.output, split_coordinate_dict[i], split_fasta_dict[i], args.pad))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    mfe_dict = {}
    for i in tqdm(range(args.threads), desc = "Merging"):
        with open(f"{args.output}/temp/mfe_{i}.pkl", "rb") as f:
            data = pickle.load(f)
        mfe_dict.update(data)
    with open(f"{args.output}/mfe.pkl", "wb") as f:
        pickle.dump(mfe_dict, f)
    printmessage("End")
    return None


def main_2():
    printmessage("Start")
    args = parse_args()

    os.makedirs(args.output, exist_ok = True)
    fasta_dict = get_fasta_as_dict(args.fasta)
    group_dict = {"FN":f"{args.coordinate}/FN.pos.txt",
                  "FP":f"{args.coordinate}/FP.pos.txt",
                  "TP":f"{args.coordinate}/TP.pos.txt"}

    for group, path in group_dict.items():
        os.makedirs(f"{args.output}/temp", exist_ok = True)
        coordinate_keys = get_keys_from_tsv(path)
        coordinate_group = group_coordinates(coordinate_keys)
        split_fasta_dict, split_coordinate_dict = split_fasta_and_coordinate(coordinate_group, fasta_dict, threads = args.threads)
        proc_list = []
        for i in range(args.threads):
            proc = mp.Process(target = get_mfe, args = (i, args.output, split_coordinate_dict[i], split_fasta_dict[i], args.pad))
            proc_list.append(proc)
            proc.start()
        for proc in proc_list:
            proc.join()
        mfe_dict = {}
        for i in tqdm(range(args.threads), desc = "Merging"):
            with open(f"{args.output}/temp/mfe_{i}.pkl", "rb") as f:
                data = pickle.load(f)
            mfe_dict.update(data)
        with open(f"{args.output}/mfe_pad{args.pad}_{group}.pkl", "wb") as f:
            pickle.dump(mfe_dict, f)
        shutil.rmtree(f"{args.output}/temp")

    printmessage("End")
    return None


def plot(path, pad):
    plt.rcParams.update({'font.size': 18})

    path_dict = {"FN": f"{path}/mfe_pad{pad}_FN.pkl",
                 "FP": f"{path}/mfe_pad{pad}_FP.pkl",
                 "TP": f"{path}/mfe_pad{pad}_TP.pkl"}
    mfe_dict = {}
    groups = list(path_dict.keys())
    for group, subpath in path_dict.items():
        with open(subpath, "rb") as f:
            data = pickle.load(f)
        data = [x for x in data.values() if x is not None]
        mfe_dict[group] = data

        print(f"{group} mean: {np.mean(data)}")
        print(f"{group} std: {np.std(data)}")
        print(f"{group} median: {np.median(data)}")
        print()


    ## Plot KDE
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    colours = ["royalblue", "tomato", "forestgreen"]
    mfe_list = [mfe_dict[x] for x in mfe_dict.keys()]
    df = pd.DataFrame({"MFE": np.concatenate(mfe_list),
                       "group": np.concatenate([[x]*len(mfe_dict[x]) for x in mfe_dict.keys()])})
    df["group"] = df["group"].apply(lambda x: f"{x} (n={len(mfe_dict[x])})")
    sns.kdeplot(data = df, x = "MFE", hue = "group", fill = False, palette = colours,
                common_norm = False, common_grid = True, legend = True, ax = ax)

    ## Run KS test
    from scipy.stats import ks_2samp
    for i in range(3):
        for j in range(i+1, 3):
            print(f"KS Test: {groups[i]} vs {groups[j]}")
            print(ks_2samp(mfe_dict[groups[i]], mfe_dict[groups[j]]))
            print()

    fig.savefig(f"{path}/mfe_pad{pad}.png")
    return None


if __name__ == "__main__":
    # main()
    # main_2()
    plot("/extdata4/baeklab/Hyeonseo/m6A/postprocess/mfe", 50)
    # plot("/extdata4/baeklab/Hyeonseo/m6A/postprocess/mfe", 50)