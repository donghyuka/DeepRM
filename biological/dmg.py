import pandas as pd
import numpy as np
import argparse
import os
import tqdm
import glob
import gc
import multiprocessing as mp
import matplotlib.pyplot as plt
from utils.utils import parse_refflat

## Differentially Methylated Gene Analysis

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--cell1", "-c1", type=str, required=True, help="Directory containing Cell 1 data")
    args.add_argument("--cell2", "-c2", type=str, required=True, help="Directory containing Cell 2 data")
    args.add_argument("--output", "-o", type=str, required=True, help="Output path")
    args.add_argument("--cpu", type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args.add_argument("--top", type=int, default=100, help="Number of top genes to select")
    args = args.parse_args()
    os.makedirs(args.output, exist_ok=True)
    return args


def worker(df, collect_list):
    gene_list = []
    ce_list = []
    epsilon = 1e-6

    for gene, gene_df in tqdm.tqdm(df, total=len(df)):
        arr_x = gene_df["cell1_pred_dom"].to_numpy()
        arr_y = gene_df["cell2_pred_dom"].to_numpy()
        if (np.sum(arr_x) < 1) and (np.sum(arr_y) < 1):
            continue
        if len(gene_df) < 5:
            continue
        if gene_df["cell1_depth"].min() < 10 or gene_df["cell2_depth"].min() < 10:
            continue
        cross_entropy = -np.mean(arr_x * np.log(arr_y + epsilon) + (1 - arr_x) * np.log(1 - arr_y + epsilon))
        gene_list.append(gene)
        ce_list.append(cross_entropy)

    collect_list.append(pd.DataFrame({"gene": gene_list, "cross_entropy": ce_list}))
    return None


def worker_bin(df, collect_list):
    gene_list = []
    ce_list = []
    epsilon = 1e-6

    for gene, gene_df in tqdm.tqdm(df, total=len(df)):
        arr_x = gene_df["cell1_pred_dom"].to_numpy() > 0
        arr_y = gene_df["cell2_pred_dom"].to_numpy() > 0
        if (np.sum(arr_x) < 1) and (np.sum(arr_y) < 1):
            continue
        if len(gene_df) < 5:
            continue
        if gene_df["cell1_depth"].min() < 10 or gene_df["cell2_depth"].min() < 10:
            continue
        cross_entropy = -np.mean(arr_x * np.log(arr_y + epsilon) + (1 - arr_x) * np.log(1 - arr_y + epsilon))
        gene_list.append(gene)
        ce_list.append(cross_entropy)

    collect_list.append(pd.DataFrame({"gene": gene_list, "cross_entropy": ce_list}))
    return None


def plotter_v1(cell_df_top, output, geneid_table = None):
    plt.rcParams.update({'font.size': 18})
    for gene, gene_df in cell_df_top:
        if geneid_table is not None:
            gene_id = geneid_table[geneid_table["transcript_id"] == gene]["gene_id"].values[0]
        else:
            gene_id = ""

        cell_1_arr = gene_df["cell1_pred_dom"].to_numpy()
        cell_2_arr = gene_df["cell2_pred_dom"].to_numpy()
        cell_2_arr = cell_2_arr * -1
        pos_arr = gene_df["pos"].to_numpy()

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

        ## use fixed bar width in pixels

        # ax.bar(pos_arr, cell_1_arr,label="HEK293T", width=3, color = plt.get_cmap("Blues")(cell_1_arr))
        # ax.bar(pos_arr, cell_2_arr, label="HeLa", width=3, color = plt.get_cmap("Reds")(cell_2_arr))

        ## use fixed bar width in pixel

        x_min = 0
        x_max = np.max(pos_arr)//100 * 100 + 100

        if x_max - x_min > 4000:
            x_grid = 400

        elif x_max - x_min > 2000:
            x_grid = 200

        else:
            x_grid = 100

        ax.set_xticks(range(x_min, x_max, x_grid))
        ax.set_xticklabels([str(x) for x in range(x_min, x_max, x_grid)])
        ax.set_xlim(x_min - 10, x_max + 10)
        ax.set_ylim(-1, 1)
        ax.set_yticks(list(np.arange(-1,1.1,0.25)))
        ax.set_yticklabels([f"{x:.2f}" for x in np.arange(-1,1.1,0.25)])
        ax.grid()

        barwidth = 0.01 * (x_max - x_min + 20)

        ax.bar(pos_arr, cell_1_arr, label="HEK293T", color="royalblue", width=barwidth)
        ax.bar(pos_arr, cell_2_arr, label="HeLa", color="tomato", width=barwidth)

        ax.set_title(f"{gene_id} ({gene}) m6A Profile")
        ax.set_xlabel("Position (nt)")
        ax.set_ylabel("Degree of Modification")
        ax.legend()

        ## hline at y=0
        ax.axhline(0, color="black", lw=3)

        plt.savefig(f"{output}/{gene}_{gene_id}.png")
        plt.close()

    return None


def plotter(cell_df_top, output, geneid_table = None):
    plt.rcParams.update({'font.size': 18})
    refflat_df = parse_refflat()
    for gene, gene_df in cell_df_top:
        if geneid_table is not None:
            gene_id = geneid_table[geneid_table["transcript_id"] == gene]["gene_id"].values[0]
        else:
            gene_id = ""

        try:
            refflat_row = refflat_df.loc[gene]
        except KeyError:
            continue
        if len(refflat_row) == 0:
            continue
        if type(refflat_row) == pd.DataFrame:
            refflat_row = refflat_row.iloc[0]

        exonstarts = refflat_row["exonStarts"]
        exonends = refflat_row["exonEnds"]

        cell_1_arr = gene_df["cell1_pred_dom"].to_numpy()
        cell_2_arr = gene_df["cell2_pred_dom"].to_numpy()
        cell_2_arr = cell_2_arr * -1
        pos_arr = gene_df["pos"].to_numpy()

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

        ## use fixed bar width in pixels

        # ax.bar(pos_arr, cell_1_arr,label="HEK293T", width=3, color = plt.get_cmap("Blues")(cell_1_arr))
        # ax.bar(pos_arr, cell_2_arr, label="HeLa", width=3, color = plt.get_cmap("Reds")(cell_2_arr))

        ## use fixed bar width in pixel

        x_min = 0
        x_max = np.max(pos_arr)//100 * 100 + 100

        if x_max - x_min > 4000:
            x_grid = 400

        elif x_max - x_min > 2000:
            x_grid = 200

        else:
            x_grid = 100

        ax.set_xticks(range(x_min, x_max, x_grid))
        ax.set_xticklabels([str(x) for x in range(x_min, x_max, x_grid)])
        ax.set_xlim(x_min - 10, x_max + 10)
        ax.set_ylim(-1, 1)
        ax.set_yticks(list(np.arange(-1,1.1,0.25)))
        ax.set_yticklabels([f"{x:.2f}" for x in np.arange(-1,1.1,0.25)])
        ax.grid()

        barwidth = 0.01 * (x_max - x_min + 20)

        ax.bar(pos_arr, cell_1_arr, label="HEK293T", color="royalblue", width=barwidth)
        ax.bar(pos_arr, cell_2_arr, label="HeLa", color="tomato", width=barwidth)

        for start, end in zip(exonstarts, exonends):
            ## Draw rectangle with rounded corners
            rect = plt.Rectangle((start, -0.05), end-start, 0.1, color="black")
            ax.add_patch(rect)


        ax.set_title(f"{gene_id} ({gene}) m6A Profile")
        ax.set_xlabel("Position (nt)")
        ax.set_ylabel("Degree of Modification")
        ax.legend()

        ## hline at y=0
        ax.axhline(0, color="black", lw=3)

        plt.savefig(f"{output}/{gene}_{gene_id}.png")
        plt.close()

    return None


def main():
    args = parse_args()
    cell1_df = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{args.cell1}/*.pkl")], axis=0)
    cell2_df = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{args.cell2}/*.pkl")], axis=0)
    cell1_df = cell1_df[["label_id", "pred_dom", "depth"]]
    cell2_df = cell2_df[["label_id", "pred_dom", "depth"]]
    cell1_df.rename(columns={"pred_dom": "cell1_pred_dom", "depth": "cell1_depth"}, inplace=True)
    cell2_df.rename(columns={"pred_dom": "cell2_pred_dom", "depth": "cell2_depth"}, inplace=True)
    cell_df = cell1_df.merge(cell2_df, on="label_id", how="outer")
    del cell1_df, cell2_df
    gc.collect()
    cell_df.fillna(0, inplace=True)

    print(cell_df)
    main_common(cell_df, args)

    return None



def main_selected():
    args = parse_args()
    # cell1_df = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{args.cell1}/*.pkl")], axis=0)
    # cell2_df = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{args.cell2}/*.pkl")], axis=0)
    # cell1_df = cell1_df[["label_id", "pred_dom", "depth"]]
    # cell2_df = cell2_df[["label_id", "pred_dom", "depth"]]
    # cell1_df.rename(columns={"pred_dom": "cell1_pred_dom", "depth": "cell1_depth"}, inplace=True)
    # cell2_df.rename(columns={"pred_dom": "cell2_pred_dom", "depth": "cell2_depth"}, inplace=True)
    # cell_df = cell1_df.merge(cell2_df, on="label_id", how="outer")
    # del cell1_df, cell2_df
    # gc.collect()
    # cell_df.fillna(0, inplace=True)
    # cell_df["gene"] = cell_df["label_id"].apply(lambda x: x.split(":")[0])
    # cell_df["pos"] = cell_df["label_id"].apply(lambda x: x.split(":")[1]).astype(int)
    #
    selected_path = "/extdata4/baeklab/Hyeonseo/m6A/biological/dmg_selected"
    top_genes = os.listdir(selected_path)
    top_genes = [x.split(".")[0] for x in top_genes]
    #
    # cell_df = cell_df[cell_df["gene"].isin(top_genes)]
    # print(cell_df)
    # cell_df.to_pickle(f"{args.output}/cell_df.pkl")
    cell_df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/biological/dmg_v2/cell_df.pkl")
    cell_df = cell_df.sort_values("pos", ascending=True).groupby("gene")

    geneid_table = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/res/ref/GRCh38_latest_genomic.convert_table.pkl")

    cell_df_top = [(gene,cell_df.get_group(gene)) for gene in top_genes]
    plotter(cell_df_top, args.output, geneid_table)


    return None

def parse_args_dry():
    args = argparse.ArgumentParser()
    args.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/biological/dryrun/dmg_out", help="Output path")
    args.add_argument("--cpu", type=int, default=64, help="Number of CPUs")
    args.add_argument("--top", type=int, default=10, help="Number of top genes to select")
    args = args.parse_args()
    return args

def main_dry():
    args = parse_args_dry()

    cell1_df = pd.concat([pd.read_pickle(f) for f in glob.glob(f"/extdata4/baeklab/Hyeonseo/m6A/biological/dryrun/dryrun_small/*.pkl")], axis=0)

    cell2_df = cell1_df.copy()

    cell2_df["noise"] = np.random.normal(0, 0.2, cell2_df.shape[0])
    cell2_df["pred_dom"] = cell2_df["pred_dom"] + cell2_df["noise"]
    cell2_df["pred_dom"] = np.clip(cell2_df["pred_dom"], 0, 1)

    cell1_df = cell1_df[["label_id", "pred_dom"]]
    cell2_df = cell2_df[["label_id", "pred_dom"]]
    cell1_df.rename(columns={"pred_dom": "cell1_pred_dom"}, inplace=True)
    cell2_df.rename(columns={"pred_dom": "cell2_pred_dom"}, inplace=True)
    cell_df = cell1_df.merge(cell2_df, on="label_id", how="outer")
    del cell1_df, cell2_df
    gc.collect()

    print(cell_df)
    main_common(cell_df, args)


    return None


def main_common(cell_df, args):
    cell_df.fillna(0, inplace=True)
    cell_df["gene"] = cell_df["label_id"].apply(lambda x: x.split(":")[0])
    cell_df["pos"] = cell_df["label_id"].apply(lambda x: x.split(":")[1]).astype(int)

    cell_df.to_pickle(f"{args.output}/cell_df.pkl")

    cell_df = cell_df.sort_values("pos", ascending=True).groupby("gene")
    cell_df_split = [[] for _ in range(args.cpu)]
    for idx, (gene, gene_df) in tqdm.tqdm(enumerate(cell_df), total=len(cell_df)):
        cell_df_split[idx%args.cpu].append((gene, gene_df))

    man = mp.Manager()
    collect_list = man.list()
    proc_list = []

    for idx, df in enumerate(cell_df_split):
        p = mp.Process(target=worker, args=(df, collect_list))
        p.start()
        proc_list.append(p)

    for p in proc_list:
        p.join()

    collect_list = list(collect_list)
    man.shutdown()

    collect_df = pd.concat(collect_list, axis=0)
    collect_df = collect_df.sort_values("cross_entropy", ascending=False)
    collect_df.to_csv(f"{args.output}/cross_entropy.tsv", sep="\t", index=False)

    top_genes = collect_df.head(args.top)["gene"]
    cell_df_top = [(gene,cell_df.get_group(gene)) for gene in top_genes]

    plotter(cell_df_top, args.output)

    return None


if __name__=="__main__":
    main_selected()



