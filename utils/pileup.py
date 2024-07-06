import multiprocessing as mp
import os
import pandas as pd
import tqdm
import argparse
import glob
import numpy as np
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", "-c", type=int, default=None, help="Number of CPUs to use")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--neg", type=float, default=0.10, help="Negative threshold")
    parser.add_argument("--pos", type=float, default=0.98, help="Positive threshold")
    parser.add_argument("--dom_neg", type=float, default=0.20, help="Dominant negative threshold")
    parser.add_argument("--dom_pos", type=float, default=0.60, help="Dominant positive threshold")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Epsilon value")
    parser.add_argument("--percentile", type=bool, default=False, help="Use percentile threshold")
    args = parser.parse_args()
    if args.cpu is None:
        args.cpu = int (0.9 * mp.cpu_count())
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/temp", exist_ok=True)
    return args


def worker(pid, file_paths, out_path, file_per_worker, threshold_neg = 0.10, threshold_pos = 0.98,
           threshold_dom_neg = 0.20, threshold_dom_pos = 0.60, epsilon = 1e-6, percentile = False):

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
        data_df["pm6a"] = np.log10(1 - np.clip(data_df["pred"].to_numpy(), 0.0, 1 - epsilon))

        if percentile:
            neg_score_array = data_df[data_df["pred"] < 0.5]["pred"].to_numpy()
            pos_score_array = data_df[data_df["pred"] >= 0.5]["pred"].to_numpy()
            threshold_neg = np.quantile(neg_score_array, 1-threshold_neg)
            threshold_pos = np.quantile(pos_score_array, threshold_pos)
            threshold_dom_neg = np.quantile(neg_score_array, 1-threshold_dom_neg)
            threshold_dom_pos = np.quantile(pos_score_array, threshold_dom_pos)

        if threshold_neg > threshold_pos:
            raise ValueError("Negative threshold must be less than positive threshold")


        data_df_bin = data_df[(data_df["pred"] <= threshold_neg) | ( data_df["pred"] >= threshold_pos)].copy()
        data_df_dom = data_df[(data_df["pred"] <= threshold_dom_neg) | ( data_df["pred"] >= threshold_dom_pos)].copy()
        data_df_dom["dom"] = data_df_dom["pred"].apply(lambda x: 1 if x >=threshold_dom_pos else 0)

        ## groupby label_id
        data_df_bin = data_df_bin.groupby("label_id").agg({"pm6a": "sum", "count": "sum"}).reset_index()
        data_df_dom = data_df_dom.groupby("label_id").agg({"dom": "sum", "pred": "sum", "count": "sum"}).reset_index()
        data_df_bin.rename(columns = {"count": "count_pm6a"}, inplace = True)
        data_df_dom.rename(columns = {"count": "count_dom", "pred": "mean_pred"}, inplace = True)
        data_df = data_df_bin.merge(data_df_dom, on = "label_id", how = "outer")
        del data_df_bin, data_df_dom
        gc.collect()

        data_df.fillna(0, inplace = True)
        fileid = pid * file_per_worker + idx
        data_df.to_pickle(f"{out_path}/temp/pileup_temp_{fileid}.pkl")

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
                                               args.dom_pos, args.epsilon, args.percentile))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    gc.collect()

    final_df = []
    for path in glob.glob(f"{args.output}/temp/pileup_temp_*.pkl"):
        final_df.append(pd.read_pickle(path))
    final_df = pd.concat(final_df, axis=0)

    print(final_df)

    final_df = final_df.groupby("label_id").agg({"pm6a": "sum", "count_pm6a": "sum",
                                                 "dom": "sum", "count_dom": "sum",
                                                 "mean_pred": "sum"}).reset_index()
    final_df["pm6a"] = final_df["pm6a"] / final_df["count_pm6a"]
    final_df["dom"] = final_df["dom"] / final_df["count_dom"]
    final_df["mean_pred"] = final_df["mean_pred"] / final_df["count_dom"]
    final_df["pm6a"] = (1 - 10**np.clip(final_df["pm6a"].to_numpy(), None, 0))
    ## convert nan to 0
    final_df.fillna(0, inplace = True)

    final_df.to_pickle(f"{args.output}/pileup.pkl")

    ## delete temp files
    for path in glob.glob(f"{args.output}/temp/pileup_temp_*.pkl"):
        os.remove(path)

    print(final_df)
    return None


if __name__ == "__main__":
    main()