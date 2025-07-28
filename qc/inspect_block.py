
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import argparse
from utils.utils import printmessage
import itertools as it

plt.style.use('default')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 22, 'legend.facecolor': 'white', 'legend.framealpha': 1, "legend.frameon": 1, "lines.linewidth": 2})


def seq_to_onehot(seq: str):
    """
    Converts a nucleotide sequence to a one-hot encoded matrix.

    Args:
        seq (str): Nucleotide sequence (A, C, G, T/U).

    Returns:
        np.ndarray: One-hot encoded matrix of the sequence.
    """
    seq = seq.upper()
    seq = seq.replace('T', 'U')
    mapping = dict(zip("ACGU", range(4)))
    mapped = [mapping[i] for i in seq]
    result = np.eye(4)[mapped].astype(float)
    return result


def motif_cdf(block_df_dict, color_dict, output):

    motif_cdf_dict = {}
    for block_name, block_df in block_df_dict.items():
        motif_arr = block_df["motif"].apply(lambda x: x[8:13]).value_counts()
        motif_cnt_arr = np.array([motif_arr.get(motif,0) for motif in motif_arr.index])
        motif_cnt_sorted = np.sort(motif_cnt_arr)[::-1]
        motif_cdf = np.cumsum(motif_cnt_sorted)/np.sum(motif_cnt_sorted)
        motif_cdf_dict[block_name] = motif_cdf

    with open(f"{output}/motif_cdf.pkl", "wb") as f:
        pickle.dump(motif_cdf_dict, f)

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,20))
    for block_name, block_df in block_df_dict.items():
        ax.plot(motif_cdf_dict[block_name], label=f"{block_name} (n={len(block_df):,})",
                color=color_dict[block_name])
    ax.set_title("5-mer Motif CDF")
    ax.set_xlabel("Motif")
    ax.set_ylabel("CDF")
    ax.legend()
    plt.savefig(f"{output}/motif_cdf.png", dpi=300)
    return None


def motif_composition(block_df_dict, color_dict,  output):
    ## Plot ratio of nucleotides in each position
    ## Each nucleotide is represented as a box, and the height of the box is the ratio of the nucleotide

    for block_name, block_df in block_df_dict.items():
        motif = block_df["motif"].apply(lambda x: seq_to_onehot(x))
        motif_sum = np.sum(motif.to_numpy(), axis=0)
        motif_sum = motif_sum/np.sum(motif_sum)
        motif_sum = pd.DataFrame(motif_sum, columns=["A","C","G","U"])

        with open(f"{output}/motif_composition.pickle", "wb") as f:
            pickle.dump([motif_sum], f)

        fig, ax = plt.subplots(1, 1, figsize=(20,20))

        motif_sum.plot(kind="bar", stacked=True, ax=ax, color=["royalblue", "tomato", "forestgreen", "gold"])
        ax.set_title(f"Motif composition: ({block_name}) (n={len(block_df):,})")
        ax.set_xlabel("Position")
        ax.set_ylabel("Ratio")
        ax.legend()
        plt.savefig(f"{output}/motif_composition_{block_name.replace(' ','_')}.png", dpi=300)

    return None


def nucleotide_composition(block_df_dict, color_dict,  output):
    ## Plot ratio of nucleotides in pie chart
    nrows =  2
    ncols =  int(np.ceil(len(block_df_dict)/nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 8*nrows))
    for i, (block_name, block_df) in enumerate(block_df_dict.items()):
        ax = axes[i//ncols, i%ncols]
        motif = block_df["motif"].apply(lambda x: seq_to_onehot(x))
        motif_sum = motif.to_numpy().sum(axis=0)
        motif_sum = np.concatenate([motif_sum[:motif_sum.shape[0]//2], motif_sum[motif_sum.shape[0]//2+1:]]).sum(axis=0)
        motif_sum = motif_sum/np.sum(motif_sum)
        ax.pie(motif_sum, labels=["A","C","G","U"], autopct='%1.1f%%')
        ax.set_title(f"{block_name}")
        ax.set_ylabel("")
        ax.set_xlabel("")

    plt.savefig(f"{output}/nucleotide_composition.png", dpi=300)
    plt.close()

    return None


def bq_plot(block_df_dict, color_dict, output, sample=int(1e+4), comment=""):
    ## Plot the distribution of base quality. Plot position-wise mean with CI95.

    stat_dict = {}
    for block_name, block_df in block_df_dict.items():
        if sample is not None:
            block_df = block_df.sample(min(sample, len(block_df))).copy()
        bq_arr = np.stack(block_df["bq"].values, axis=0)
        bq_mean = np.mean(bq_arr, axis=0)
        bq_std = np.std(bq_arr, axis=0)
        bq_ci95 = 1.96*bq_std/np.sqrt(len(bq_arr))
        stat_dict[block_name] = (bq_mean, bq_ci95)

    with open(f"{output}/bq_plot.pickle", "wb") as f:
        pickle.dump(stat_dict, f)

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,10))
    for block_name, block_df in block_df_dict.items():
        pos_bq_mean, pos_bq_ci95 = stat_dict[block_name]
        ax.plot(pos_bq_mean, color=color_dict[block_name], label=f"{block_name} (n={len(block_df):,})")
        ax.fill_between(np.arange(len(pos_bq_mean)), pos_bq_mean-pos_bq_ci95, pos_bq_mean+pos_bq_ci95,
                        color=color_dict[block_name], alpha=0.3)
    ax.set_title("Base Quality Distribution")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean Base Quality")
    ax.legend()
    plt.savefig(f"{output}/bq_plot{comment}.png", dpi=300)
    return None


def block_score_distribution(block_df_dict, color_dict, output):
    ## Plot the distribution of block score
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(20,20))
    for block_name, block_df in block_df_dict.items():
        if "block_score" in block_df.columns:
            block_df["score"] = block_df["block_score"]
        sns.histplot(block_df["score"], ax=ax, label=f"{block_name} (n={len(block_df):,})",
                     color=color_dict[block_name], linewidth=5, element="step", stat="density",
                     fill=False, binwidth = 5)
    ax.set_title("Block Score Distribution")
    ax.set_xlabel("Block Score")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(f"{output}/block_score_distribution.png", dpi=300)
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, required=True, help="Output prefix")
    parser.add_argument("--intermediate", "-i", type=str, nargs="+", default = None,  help="Intermediate files prefix")
    parser.add_argument("--score", "-p", type=int, default=100, help="Score cutoff")
    parser.add_argument("--block", "-k", type=str, required=True, nargs="+", help="Block file")
    parser.add_argument("--name", "-n", type=str, default = None, nargs="+", help="Block name")
    parser.add_argument("--type", "-t", type=str, required=True, nargs="+", help="Block type")
    parser.add_argument("--sample", "-s", type=int, default=int(1e+6), help="Sampling fraction")
    parser.add_argument("--cb_len", "-c", type=int, default=41, help="Context block length")
    args = parser.parse_args()
    assert len(args.block) == len(args.type)
    assert all([os.path.exists(b) for b in args.block])
    if args.intermediate is not None:
        assert len(args.intermediate) == len(args.block)
        assert all([os.path.exists(i) for i in args.intermediate])
    else:
        intermediate = []
        for path in args.block:
            parent_dir = os.path.dirname(os.path.dirname(path))
            intermediate.append(f"{parent_dir}/qc/block_df_dict.pkl")
        args.intermediate = intermediate
    if args.name is not None:
        assert len(args.name) == len(args.block)
    else:
        name = []
        for path in args.block:
            name_candidate = set([x for x in path.split("/") if (x.startswith("ON") and x[2:].isdigit())])
            if len(name_candidate) == 1:
                name.append(name_candidate.pop())
            else:
                raise ValueError("Unable to detect name from block file. Supply --name manually.")
        args.name = name
        printmessage("Names detected:",args.name)
    return args

def plot_violin(block_df_dict, color_dict, cb_len, output):
    ## merge df
    df_list = []
    color_list = []
    for name, df in block_df_dict.items():
        df = df[["bq"]].copy()
        # df = df.sample(frac=0.1)
        bq_idx = range(cb_len)
        df["bq_idx"] = np.tile(bq_idx, (len(df),1)).tolist()
        df = df.explode(["bq", "bq_idx"])
        df["name"] = name
        df["bq"] = df["bq"].astype(int)
        df["bq_idx"] = df["bq_idx"].astype(int)
        df = df.dropna()
        df_list.append(df)
        print(df)
        color_list.append(color_dict[name])

    palette = sns.color_palette(color_list)

    df = pd.concat(df_list).reset_index(drop=True)
    print(df)

    fig, ax = plt.subplots(1, 1, figsize=(30,10))
    sns.violinplot(x="bq_idx", y="bq", hue="name", data=df, ax=ax, palette=palette, linewidth=0.5,
                   inner=None, hue_order=list(block_df_dict.keys()))
    ax.set_title("Base Quality Distribution")
    ax.set_xlabel("Position")
    ax.set_ylabel("Base Quality")
    ax.legend()
    plt.savefig(f"{output}/bq_violin.png", dpi=300)

    return None



def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    warm_color_list = it.cycle(["tomato", "coral", "orange", "gold", "goldenrod", "chocolate"])
    cool_color_list = it.cycle(["royalblue", "dodgerblue", "deepskyblue", "skyblue", "lightblue", "powderblue"])
    modified_name_list = ["m6A", "m1A", "Am", "I",
                          "m5C", "hm5C", "Cm",
                          "m7G", "m1G", "Gm",
                          "m5U", "Um", "pseU"]

    block_df_dict = {}
    perfect_block_df_dict = {}
    color_dict = {}

    for block, name, block_type in zip(args.block, args.name, args.type):
        block_name = f"{name} ({block_type})"
        if block_type in modified_name_list:
            color = next(warm_color_list)
        else:
            color = next(cool_color_list)
        color_dict[block_name] = color

    load_success = False
    output_file_exists = False

    if os.path.exists(args.output):
        printmessage("Output directory already exists. Attempting to load pickle.")
        try:
            block_df_dict = pickle.load(open(f"{args.output}/block_df_dict.pkl", "rb"))
            perfect_block_df_dict = pickle.load(open(f"{args.output}/perfect_block_df_dict.pkl", "rb"))
            load_success = True
            output_file_exists = True
            printmessage("Pickle loading successful.")
        except:
            printmessage("Pickle loading from output directory failed.")
            load_success = False
            output_file_exists = False
            block_df_dict = {}
            perfect_block_df_dict = {}

    if not load_success:
        printmessage("Attempting to load intermediate files.")
        try:
            for intermediate in args.intermediate:
                block_df_dict_run = pickle.load(open(intermediate, "rb"))
                perfect_block_df_dict_run = pickle.load(open(intermediate.replace("block_df_dict", "perfect_block_df_dict"), "rb"))
                block_df_dict.update(block_df_dict_run)
                perfect_block_df_dict.update(perfect_block_df_dict_run)
            load_success = True
            output_file_exists = False
            printmessage("Pickle loading successful.")
        except:
            printmessage("Pickle loading failed.")
            load_success = False
            output_file_exists = False
            block_df_dict = {}
            perfect_block_df_dict = {}

    if not load_success:
        for block, name, block_type in zip(args.block, args.name, args.type):
            block_name = f"{name} ({block_type})"
            block_df = pd.read_pickle(block)
            perfect_block_df = block_df[block_df["score"] >= args.score]
            perfect_block_df = perfect_block_df.sample(min(args.sample, len(perfect_block_df))).copy()
            perfect_block_df_dict[block_name] = perfect_block_df
            block_df = block_df.sample(min(args.sample, len(block_df))).copy()
            block_df_dict[block_name] = block_df

    if not output_file_exists:
        with open(f"{args.output}/block_df_dict.pkl", "wb") as f:
            pickle.dump(block_df_dict, f)

        with open(f"{args.output}/perfect_block_df_dict.pkl", "wb") as f:
            pickle.dump(perfect_block_df_dict, f)

    nucleotide_composition(perfect_block_df_dict, color_dict, args.output)
    motif_composition(perfect_block_df_dict, color_dict, args.output)
    bq_plot(perfect_block_df_dict, color_dict, args.output)
    plot_violin(perfect_block_df_dict, color_dict, args.cb_len, args.output)
    motif_cdf(perfect_block_df_dict, color_dict, args.output)
    block_score_distribution(block_df_dict, color_dict, args.output)

    return None


def plot_motif(perfect_block_df_dict, color_dict, args, motif_list = ["AGACU","CGACA","UGAUC","GAAGC","UCAAG"]):

    for block_name, block_df in perfect_block_df_dict.items():
        block_df["motif"] = block_df["motif"].apply(lambda x: x[8:13])
        perfect_block_df_dict[block_name] = block_df

    for motif in motif_list:
        motif_block_df_dict= {}
        for block_name, block_df in perfect_block_df_dict.items():
            motif_block_df = block_df[block_df["motif"] == motif].copy()
            motif_block_df_dict[block_name] = motif_block_df
        print(motif_block_df_dict)
        bq_plot(motif_block_df_dict, color_dict, args.output, sample=None, comment=f"-{motif}")
        plot_violin(perfect_block_df_dict, color_dict, args.cb_len, args.output)

    return None



if __name__ == "__main__":
    main()