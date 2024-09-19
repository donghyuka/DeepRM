import multiprocessing as mp
import os
import pandas as pd
import tqdm
import argparse
import glob
import numpy as np
import gc
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", "-c", type=int, default=None, help="Number of CPUs to use")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--neg", type=float, default=0.10, help="Negative threshold")
    parser.add_argument("--pos", type=float, default=0.98, help="Positive threshold")
    parser.add_argument("--dom_neg", type=float, default=0.20, help="Dominant negative threshold")
    parser.add_argument("--dom_pos", type=float, default=0.60, help="Dominant positive threshold")
    parser.add_argument("--epsilon", type=float, default=1e-30, help="Epsilon value")
    parser.add_argument("--norm", action = "store_true", help="Use depth normalisation")
    parser.add_argument("--method", type=str, default="abs", help="Method for depth normalisation")

    args = parser.parse_args()
    if args.cpu is None:
        args.cpu = int (0.9 * mp.cpu_count())
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/temp", exist_ok=True)
    return args


def worker(pid, file_paths, out_path, file_per_worker, threshold_neg = 0.10, threshold_pos = 0.98,
           threshold_dom_neg = 0.20, threshold_dom_pos = 0.60, epsilon = 1e-30):

    pileup_list = []

    for idx, path in enumerate(tqdm.tqdm(file_paths)):

        if path.endswith(".pkl"):
            data_df = pd.read_pickle(path)
        elif path.endswith(".tsv"):
            data_df = pd.read_csv(path, sep="\t")
        else:
            raise ValueError("Input file must be either .tsv or .pkl")

        # threshold = gmm_em_lda(data_df["pred"].values)

        ## Groupby label_id and get mean of predictions
        data_df["count"] = 1

        data_df_pos = data_df[data_df["pred"] >= threshold_pos].copy()
        data_df_pos["pm6a"] = np.log10(1 - np.clip(data_df_pos["pred"].to_numpy(), 0.0, 1 - epsilon))
        data_df_neg = data_df[data_df["pred"] < threshold_neg].copy()
        data_df_neg["pca"] = np.log10(np.clip(data_df_neg["pred"].to_numpy(), epsilon, 1.0))
        data_df_dom = data_df[(data_df["pred"] < threshold_dom_neg) | ( data_df["pred"] >= threshold_dom_pos)].copy()
        data_df_dom["dom"] = data_df_dom["pred"].apply(lambda x: 1 if x >=threshold_dom_pos else 0)

        ## groupby label_id
        data_df_pos = data_df_pos.groupby("label_id").agg({"pm6a": "sum", "count": "sum"}).reset_index()
        data_df_neg = data_df_neg.groupby("label_id").agg({"pca": "sum", "count": "sum"}).reset_index()
        data_df_dom = data_df_dom.groupby("label_id").agg({"dom": "sum", "count": "sum"}).reset_index()
        data_df = data_df.groupby("label_id").agg({"count": "sum"}).reset_index()
        data_df_pos.rename(columns = {"count": "count_pm6a"}, inplace = True)
        data_df_neg.rename(columns = {"count": "count_pca"}, inplace = True)
        data_df_dom.rename(columns = {"count": "count_dom"}, inplace = True)
        data_df = data_df.merge(data_df_pos, on = "label_id", how = "left")
        data_df = data_df.merge(data_df_neg, on = "label_id", how = "left")
        data_df = data_df.merge(data_df_dom, on = "label_id", how = "left")
        del data_df_pos, data_df_dom
        gc.collect()
        data_df.fillna(0, inplace = True)
        pileup_list.append(data_df)

    data_df = pd.concat(pileup_list, axis=0)

    data_df = data_df.groupby("label_id").agg({"pm6a": "sum", "count_pm6a": "sum",
                                               "pca": "sum", "count_pca": "sum",
                                               "dom": "sum", "count_dom": "sum", "count": "sum"}).reset_index()

    path = f"{out_path}/temp/pileup_temp_{pid}.npz"

    label_id = data_df["label_id"].values
    pm6a = data_df["pm6a"].values.astype(np.float32)
    count_pm6a = data_df["count_pm6a"].values.astype(np.int32)
    dom = data_df["dom"].values.astype(np.float32)
    count_dom = data_df["count_dom"].values.astype(np.int32)
    count = data_df["count"].values.astype(np.int32)
    pca = data_df["pca"].values.astype(np.float32)
    count_pca = data_df["count_pca"].values.astype(np.int32)
    np.savez_compressed(path, label_id = label_id, pm6a = pm6a, count_pm6a = count_pm6a,
                        pca = pca, count_pca = count_pca,
                        dom = dom, count_dom = count_dom, count = count)

    del data_df
    gc.collect()

    return None


def main():
    args = parse_args()
    file_paths = glob.glob(f"{args.input}/*.pkl") + glob.glob(f"{args.input}/*.tsv")
    proc_list = []
    file_paths_split = np.array_split(file_paths, args.cpu)
    file_per_worker = np.ceil(len(file_paths) / args.cpu).astype(int)

    for pid, file_paths in enumerate(file_paths_split):
        proc = mp.Process(target=worker, args=(pid, file_paths, args.output, file_per_worker,
                                               args.neg, args.pos, args.dom_neg,
                                               args.dom_pos, args.epsilon))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    gc.collect()

    keys = ["label_id", "pm6a", "count_pm6a", "pca", "count_pca", "dom", "count_dom", "count"]
    final_df = {key:[] for key in keys}
    for path in tqdm.tqdm(glob.glob(f"{args.output}/temp/pileup_temp_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            for key in keys:
                final_df[key].append(data[key])
    final_df = {key:np.concatenate(final_df[key]) for key in keys}
    final_df = pd.DataFrame(final_df)
    print(final_df)

    final_df = final_df.groupby("label_id").agg({"pm6a": "sum", "count_pm6a": "sum",
                                                 "pca": "sum", "count_pca": "sum",
                                                 "dom": "sum", "count_dom": "sum", "count": "sum"}).reset_index()

    final_df["dom"] = final_df["dom"] / final_df["count_dom"].clip(1, None)
    final_df["logsum_pm6a"] = final_df["pm6a"]
    final_df["logsum_pca"] = final_df["pca"]

    # if args.norm:
    #     if args.method == "abs":
    #         final_df["pm6a"] = final_df["pm6a"] / final_df["count"].clip(1, None)
    #     elif args.method == "eff":
    #         final_df["pm6a"] = final_df["pm6a"] / final_df["count_pm6a"].clip(1, None)
    #     elif args.method == "dom":
    #         final_df["pm6a"] = final_df["pm6a"] / (final_df["count_dom"].clip(1, None)*(1-final_df["dom"])).clip(1, None)
    #     elif args.method == "fpr":
    #         final_df["pm6a"] = final_df["pm6a"] / (1+final_df["count"] * 0.003)

    final_df["pm6a"] = (1 - 10**np.clip(final_df["pm6a"].to_numpy(), None, 0))
    final_df["pca"] = (1 - 10**np.clip(final_df["pca"].to_numpy(), None, 0))
    final_df.fillna(0, inplace = True)

    path = f"{args.output}/pileup.npz"

    label_id = final_df["label_id"].values
    pm6a = final_df["pm6a"].values.astype(np.float32)
    count_pm6a = final_df["count_pm6a"].values.astype(np.int32)
    dom = final_df["dom"].values.astype(np.float32)
    count_dom = final_df["count_dom"].values.astype(np.int32)
    count = final_df["count"].values.astype(np.int32)
    logsum_pm6a = final_df["logsum_pm6a"].values.astype(np.float32)
    pca = final_df["pca"].values.astype(np.float32)
    count_pca = final_df["count_pca"].values.astype(np.int32)
    logsum_pca = final_df["logsum_pca"].values.astype(np.float32)
    np.savez_compressed(path, label_id = label_id, pm6a = pm6a, count_pm6a = count_pm6a,
                        pca = pca, count_pca = count_pca, logsum_pca = logsum_pca,
                        dom = dom, count_dom = count_dom, count = count, logsum_pm6a = logsum_pm6a)

    # delete temp files
    shutil.rmtree(f"{args.output}/temp")

    print(final_df)
    return None


if __name__ == "__main__":
    main()