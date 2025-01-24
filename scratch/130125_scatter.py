import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
import sys
import glob
import tqdm
from scipy import stats
import multiprocessing as mp
from utils.utils import printmessage


plt.style.use('default')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 22, 'legend.facecolor': 'white', 'legend.framealpha': 1, "legend.frameon": 1, "lines.linewidth": 2})

def get_density(x,y, sample=1000, bw_method= 0.2, threads=120):
    values = np.stack([x, y], axis=0)

    kernel = stats.gaussian_kde(values, bw_method=bw_method)
    printmessage("Kernel covariance: " + str(kernel.covariance))
    values_split = np.array_split(values, threads, axis=1)
    procs = []
    man = mp.Manager()
    results = man.dict()
    for i in range(threads):
        proc = mp.Process(target=worker, args=(values_split[i], kernel, results, i))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
    density = np.concatenate([results[i] for i in range(threads)])
    man.shutdown()
    return density

def worker(values, kernel, results, i):
    results[i] = kernel(values)
    return None

def plot_scatter(df, x = "m6A_level", y = "dom", alpha=1.0, s = 30, noplot = False, bw_method = 0.2, sample = 100000, outpath = None, vmax = 1):

    if os.path.exists(outpath+".density.npy"):
        density = np.load(outpath+".density.npy")
    else:
        density = get_density(df[x], df[y], sample=sample, bw_method=bw_method)
        np.save(outpath+".density.npy", density)

    plt.style.use('default')
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({'font.size': 22, 'legend.facecolor': 'white', 'legend.framealpha': 1, "legend.frameon": 1, "lines.linewidth": 2})
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    sns.scatterplot(x=df[x], y=df[y], ax=ax, hue=density, palette="flare_r", hue_norm=(0,vmax),
                    s=s, legend=False, linewidth=0, alpha = alpha)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.aspect = 1

    if outpath:
        plt.savefig(outpath)

    printmessage("Saved to " + outpath)

    return None

def dom_threshold(df):
    return np.log10(0.02)*(df["dom"]) > df["logsum_1_p_pos"]/df["count_all"]

def main():

    hek_label_orig = pd.read_csv("/extdata3/baeklab/Jungmin/RNAmod/valid.new/cmp/glori.txt", sep="\t")
    hela_label_orig = pd.read_csv("/extdata3/baeklab/Jungmin/RNAmod/valid.new/cmp/etam.txt", sep="\t")

    new_dfs = []
    for df in [hek_label_orig, hela_label_orig]:
        df = df.copy()
        df["genome_id"] = df["chr"] + ":" + df["str"].astype(str) + ":" + df["pos"].astype(str)
        df.rename(columns={"dom": "m6A_level", "depth": "exp_depth"}, inplace=True)
        df["m6a"] = (df["qval"] < 1E-7) & ~(df["qval_neg"] < 0.1)
        df["m6A_level"] = df["m6A_level"].clip(0, 1) * (df["m6a"].fillna(0))
        df = df[~((df["qval"] < 1E-1) & (df["qval"] >= 1E-7))]
        df = df[["genome_id", "m6A_level","exp_depth"]].fillna(0).copy()
        new_dfs.append(df)
    hek_label, hela_label = new_dfs

    hek_airna_df = "/extdata4/baeklab/Hyeonseo/m6A/inference/pileup/AIRNA-DW-v4-m6A_ON0118-20241129-120647-8-182000-token_normalise_ver121224_mergedref_ON0090_m6Afinal/pileup_strid.npz.genomic.pkl"
    hek_airna_df = pd.read_pickle(hek_airna_df)
    hela_airna_df = "/extdata4/baeklab/Hyeonseo/m6A/inference/pileup/AIRNA-DW-v4-m6A_ON0118-20241129-120647-8-182000-token_normalise_ver121224_mergedref_ON0091_m6Afinal/pileup_strid.npz.genomic.pkl"
    hela_airna_df = pd.read_pickle(hela_airna_df)
    hek_airna_df = hek_airna_df.reset_index(drop=True).merge(hek_label, on="genome_id", how="inner")
    hela_airna_df = hela_airna_df.reset_index(drop=True).merge(hela_label, on="genome_id", how="inner")
    hek_airna_df["dom_threshold"] = dom_threshold(hek_airna_df)
    hela_airna_df["dom_threshold"] = dom_threshold(hela_airna_df)

    hek_m6anet_df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0090/gene_df.pkl")
    hela_m6anet_df = pd.read_pickle("/extdata4/baeklab/Hyeonseo/m6A/m6anet/ON0091/gene_df.pkl")
    hek_m6anet_df = hek_m6anet_df.reset_index(drop=True).merge(hek_label, on="genome_id", how="inner")
    hela_m6anet_df = hela_m6anet_df.reset_index(drop=True).merge(hela_label, on="genome_id", how="inner")

    df = hela_airna_df
    outdir = "/extdata4/baeklab/Hyeonseo/m6A/poster/scatter_3/hela_airna"
    plot_scatter(df[df["dom_threshold"] & (df["count_all"] > 40)], outpath = outdir + "_40.png")
    plot_scatter(df[df["dom_threshold"] & (df["count_all"] > 30) & (df["count_all"] <= 40)], s = 160, outpath = outdir + "_30.png")
    plot_scatter(df[df["dom_threshold"] & (df["count_all"] > 20) & (df["count_all"] <= 30)], s = 160, outpath = outdir + "_20.png")
    plot_scatter(df[df["dom_threshold"] & (df["count_all"] > 10) & (df["count_all"] <= 20)], s = 160, outpath = outdir + "_10.png")
    plot_scatter(df[df["dom_threshold"] & (df["count_all"] > 5) & (df["count_all"] <= 10)],  s = 160, outpath = outdir + "_5.png")

    df = hek_airna_df
    outdir = "/extdata4/baeklab/Hyeonseo/m6A/poster/scatter_3/hek_airna"
    plot_scatter(df[df["dom_threshold"] & (df["count_all"] > 40)], outpath = outdir + "_40.png")
    plot_scatter(df[df["dom_threshold"] & (df["count_all"] > 30) & (df["count_all"] <= 40)], s = 160, outpath = outdir + "_30.png")
    plot_scatter(df[df["dom_threshold"] & (df["count_all"] > 20) & (df["count_all"] <= 30)], s = 160, outpath = outdir + "_20.png")
    plot_scatter(df[df["dom_threshold"] & (df["count_all"] > 10) & (df["count_all"] <= 20)], s = 160, outpath = outdir + "_10.png")
    plot_scatter(df[df["dom_threshold"] & (df["count_all"] > 5) & (df["count_all"] <= 10)],  s = 160, outpath = outdir + "_5.png")

    df = hela_m6anet_df
    outdir = "/extdata4/baeklab/Hyeonseo/m6A/poster/scatter_3/hela_m6anet"
    plot_scatter(df[(df["pm6a"] > 0) & (df["count_dom"] > 40)], outpath = outdir + "_40.png")
    plot_scatter(df[(df["pm6a"] > 0) & (df["count_dom"] > 30) & (df["count_dom"] <= 40)], s = 160, outpath = outdir + "_30.png")
    plot_scatter(df[(df["pm6a"] > 0) & (df["count_dom"] > 20) & (df["count_dom"] <= 30)], s = 160, outpath = outdir + "_20.png")

    df = hek_m6anet_df
    outdir = "/extdata4/baeklab/Hyeonseo/m6A/poster/scatter_3/hek_m6anet"
    plot_scatter(df[(df["pm6a"] > 0) & (df["count_dom"] > 40)], outpath = outdir + "_40.png")
    plot_scatter(df[(df["pm6a"] > 0) & (df["count_dom"] > 30) & (df["count_dom"] <= 40)], s = 160, outpath = outdir + "_30.png")
    plot_scatter(df[(df["pm6a"] > 0) & (df["count_dom"] > 20) & (df["count_dom"] <= 30)], s = 160, outpath = outdir + "_20.png")

    return None

if __name__ == "__main__":
    main()