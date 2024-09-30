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
    parser.add_argument("--output", "-o", type=str, default ="/extdata4/baeklab/Hyeonseo/m6A/inference/pileup", help="Output path")
    parser.add_argument("--neg", type=float, default=0.10, help="Negative threshold")
    parser.add_argument("--pos", type=float, default=0.98, help="Positive threshold")
    parser.add_argument("--dom_neg", type=float, default=0.20, help="Dominant negative threshold")
    parser.add_argument("--dom_pos", type=float, default=0.60, help="Dominant positive threshold")
    parser.add_argument("--epsilon", type=float, default=1e-30, help="Epsilon value")
    parser.add_argument("--postfix", "-x", type=str, default="", help="Comment")

    args = parser.parse_args()
    if args.cpu is None:
        args.cpu = int (0.9 * mp.cpu_count())

    if args.input.endswith("/"):
        args.input = args.input[:-1]
    if args.output.endswith("/"):
        args.output = args.output[:-1]

    args.output = os.path.join(args.output, os.path.basename(args.input) + args.postfix)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f"{args.output}/temp", exist_ok=True)
    return args


def worker(pid, file_paths, out_path, file_per_worker, threshold_neg = 0.10, threshold_pos = 0.98,
           threshold_dom_neg = 0.20, threshold_dom_pos = 0.60, epsilon = 1e-30):

    keys = ["label_id", "p_sum", "logsum_1_p_pos", "count_all", "dom", "count_dom"]

    key_dict = {"str": ["label_id"],
                "float32": ["p_sum", "logsum_1_p_pos", "dom"],
                "int32": ["count_all", "count_dom"]}

    data_dict = {"str": [], "float32": [], "int32": []}

    for idx, path in enumerate(tqdm.tqdm(file_paths)):

        if path.endswith(".pkl"):
            data_df_all = pd.read_pickle(path)
        elif path.endswith(".tsv"):
            data_df_all = pd.read_csv(path, sep="\t")
        elif path.endswith(".npz"):
            with np.load(path, allow_pickle=True) as data:
                data_df_all = {key: data[key] for key in data.keys()}
                data_df_all = pd.DataFrame(data_df_all)
        else:
            raise ValueError("Input file must be either .tsv or .pkl")
        data_df_all = pd.DataFrame(data_df_all)
        data_df_all["count"] = 1

        assert np.min(data_df_all["pred"].values) >= 0.0, f"Minimum value of pred is {np.min(data_df_all['pred'].values)}"
        assert np.max(data_df_all["pred"].values) <= 1.0, f"Maximum value of pred is {np.max(data_df_all['pred'].values)}"

        data_df_all["logsum_1_p"] = np.log10(np.clip(1 - data_df_all["pred"].values, epsilon, 1.0))
        data_df_all["dom"] = data_df_all["pred"].apply(lambda x: 1 if x >=threshold_dom_pos else 0)

        data_df_pos = data_df_all[data_df_all["pred"] >= threshold_pos]
        data_df_dom = data_df_all[(data_df_all["pred"] < threshold_dom_neg) | (data_df_all["pred"] >= threshold_dom_pos)]

        ## groupby label_id
        data_df_pos = data_df_pos.groupby("label_id").agg({"logsum_1_p": "sum"}).reset_index()
        data_df_dom = data_df_dom.groupby("label_id").agg({"dom": "sum", "count": "sum"}).reset_index()
        data_df_all = data_df_all.groupby("label_id").agg({"count": "sum", "pred": "sum"}).reset_index()


        data_df_pos.rename(columns = {"logsum_1_p": "logsum_1_p_pos"}, inplace = True)
        data_df_all.rename(columns = {"count": "count_all", "pred":"p_sum"}, inplace = True)
        data_df_dom.rename(columns = {"count": "count_dom"}, inplace = True)

        data_df_all = data_df_all.merge(data_df_pos, on = "label_id", how = "left")
        data_df_all = data_df_all.merge(data_df_dom, on = "label_id", how = "left")

        data_df_all.fillna(0, inplace = True)

        str_array = data_df_all[key_dict["str"]].values
        float32_array = data_df_all[key_dict["float32"]].values.astype(np.float32)
        int32_array = data_df_all[key_dict["int32"]].values.astype(np.int32)

        data_dict["str"].append(str_array)
        data_dict["float32"].append(float32_array)
        data_dict["int32"].append(int32_array)

        del data_df_pos, data_df_dom, data_df_all
        gc.collect()

    data_dict = {key:np.concatenate(data_dict[key]) for key in data_dict.keys()}
    data_dict = pd.concat([pd.DataFrame(data_dict["str"], columns = key_dict["str"]),
                          pd.DataFrame(data_dict["float32"], columns = key_dict["float32"]),
                          pd.DataFrame(data_dict["int32"], columns = key_dict["int32"])],
                         axis = 1)
    gc.collect()

    data_dict = data_dict.groupby("label_id").agg({key: "sum" for key in keys if key != "label_id"}).reset_index()

    path = f"{out_path}/temp/pileup_temp_{pid}.npz"

    np.savez_compressed(path, **{key: data_dict[key].values for key in keys})

    del data_dict
    gc.collect()

    return None


def main():
    args = parse_args()
    file_paths = glob.glob(f"{args.input}/*.pkl") + glob.glob(f"{args.input}/*.tsv") + glob.glob(f"{args.input}/*.npz")
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

    keys = ["label_id", "p_sum", "logsum_1_p_pos", "count_all", "dom", "count_dom"]

    final_df = {key:[] for key in keys}
    for path in tqdm.tqdm(glob.glob(f"{args.output}/temp/pileup_temp_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            for key in keys:
                final_df[key].append(data[key])
    final_df = {key:np.concatenate(final_df[key]) for key in keys}
    final_df = pd.DataFrame(final_df)
    print(final_df)

    final_df = final_df.groupby("label_id").agg({key: "sum" for key in keys if key != "label_id"}).reset_index()
    final_df.fillna(0, inplace = True)

    final_df["dom"] = final_df["dom"] / final_df["count_dom"].clip(1, None)
    final_df["log_pm6a"] =  -(2-final_df["dom"])*final_df["logsum_1_p_pos"]/final_df["count_all"].clip(1, None)

    keys = ["label_id", "p_sum", "logsum_1_p_pos", "count_all", "count_dom", "dom", "log_pm6a"]

    path = f"{args.output}/pileup.npz"
    np.savez_compressed(path, **{key: final_df[key].values for key in keys})

    # delete temp files
    shutil.rmtree(f"{args.output}/temp")

    print(final_df)
    return None


if __name__ == "__main__":
    main()