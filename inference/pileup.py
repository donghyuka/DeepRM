import multiprocessing as mp
import os
import pandas as pd
import tqdm
import argparse
import glob
import numpy as np
import gc
import shutil
from utils.utils import reformat_transcript_id

def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", "-c", type=int, default=None, help="Number of CPUs to use")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--mpileup", "-m", type=str, required=True, help="Filtered mpileup file path")
    parser.add_argument("--pos", "-p", type=float, default=0.98, help="Positive threshold")
    parser.add_argument("--epsilon", "-e", type=float, default=1e-30, help="Epsilon value")
    parser.add_argument("--postfix", "-x", type=str, default="final", help="Comment")

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

def worker(pid, file_paths, out_path, threshold_pos = 0.98,epsilon = 1e-30, key_dict={}):
    """
    Worker function to process a subset of files.

    Args:
        pid (int): Process ID.
        file_paths (list): List of file paths to process.
        out_path (str): Output path.
        threshold_pos (float, optional): Positive threshold. Defaults to 0.98.
        epsilon (float, optional): Epsilon value. Defaults to 1e-30.
        key_dict (dict, optional): Dictionary of keys for data types. Defaults to {}.

    Returns:
        None
    """
    keys = np.concatenate(list(key_dict.values()))

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

        assert np.min(data_df_all["pred"].values) >= 0.0, f"Minimum value of pred is {np.min(data_df_all['pred'].values)}"
        assert np.max(data_df_all["pred"].values) <= 1.0, f"Maximum value of pred is {np.max(data_df_all['pred'].values)}"

        data_df_all["count_all"] = 1
        data_df_all["count_pos"] = data_df_all["pred"].apply(lambda x: 1 if x >= threshold_pos else 0)
        data_df_all["logsum_1_p_pos"] = np.log10(np.clip(1 - data_df_all["pred"].values, epsilon, 1.0)) * data_df_all["count_pos"]
        data_df_all["kl_div"] = data_df_all["pred"] * np.log2(2*data_df_all["pred"]) + (1-data_df_all["pred"])*np.log2(2*(1-data_df_all["pred"]))
        data_df_all["kl_div_neg"] = data_df_all["kl_div"] * (data_df_all["pred"] <= 0.5)
        data_df_all["kl_div_pos"] = data_df_all["kl_div"] * (data_df_all["pred"] > 0.5)
        ## groupby label_id
        data_df_all = data_df_all.groupby("label_id").agg({key: "sum" for key in keys if key != "label_id"}).reset_index().fillna(0)

        str_array = data_df_all[key_dict["str"]].values
        float32_array = data_df_all[key_dict["float32"]].values.astype(np.float32)
        int32_array = data_df_all[key_dict["int32"]].values.astype(np.int32)

        data_dict["str"].append(str_array)
        data_dict["float32"].append(float32_array)
        data_dict["int32"].append(int32_array)

        del data_df_all
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
    """
    Main function to parse arguments and process files.

    Returns:
        None
    """
    args = parse_args()
    file_paths = glob.glob(f"{args.input}/*.pkl") + glob.glob(f"{args.input}/*.tsv") + glob.glob(f"{args.input}/*.npz")
    proc_list = []
    file_paths_split = np.array_split(file_paths, args.cpu)
    file_per_worker = np.ceil(len(file_paths) / args.cpu).astype(int)

    key_dict = {"str": ["label_id"],
                "float32": ["logsum_1_p_pos", "kl_div_neg", "kl_div_pos"],
                "int32": ["count_all", "count_pos"]}

    keys = np.concatenate(list(key_dict.values()))

    for pid, file_paths in enumerate(file_paths_split):
        proc = mp.Process(target=worker, args=(pid, file_paths, args.output,
                                               args.pos, args.epsilon, key_dict))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    gc.collect()

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

    final_df["dom"] = final_df["kl_div_pos"] / (final_df["kl_div_neg"] + final_df["kl_div_pos"])
    final_df["pm6a"] = -(2-final_df["dom"])*final_df["logsum_1_p_pos"]/final_df["count_all"] + ((1-final_df["dom"])*np.log10(np.clip(1-final_df["dom"],1e-30,1)) + final_df["dom"] * np.log10(np.clip(final_df["dom"],1e-30,1)))*(final_df["count_pos"]/final_df["count_all"])

    keys = ["label_id", "pm6a", "dom", "count_all", "count_pos", "kl_div_neg", "kl_div_pos", "logsum_1_p_pos"]

    path = f"{args.output}/pileup.npz"
    np.savez_compressed(path, **{key: final_df[key].values for key in keys})

    # delete temp files
    shutil.rmtree(f"{args.output}/temp")

    with np.load(f"{args.output}/pileup.npz", allow_pickle=True) as data:
        final_df = {key:data[key] for key in data.keys()}
    final_df = pd.DataFrame(final_df)
    print(final_df)


    if isinstance(final_df["label_id"][0],str):
        if final_df["label_id"][0].isnumeric():
            reindex_flag = True
        else:
            reindex_flag = False
    else:
        reindex_flag = True

    if reindex_flag:
        if args.mpileup.endswith(".pkl"):
            label_df = pd.read_pickle(args.mpileup)
        elif args.mpileup.endswith(".tsv"):
            label_df = pd.read_csv(args.mpileup, sep="\t")
        else:
            raise ValueError("Filtered mpileup file must be either .tsv or .pkl")


        label_df["index"] = label_df.index
        label_df["ref"] = label_df["ref"].apply(reformat_transcript_id)
        label_df["label_id"] = label_df["ref"] + ":" + (label_df["pos"]-1).astype(str)

        index_id_dict = dict(zip(label_df["index"], label_df["label_id"]))
        del label_df
        gc.collect()

        final_df["label_id"] = final_df["label_id"].map(index_id_dict)

        path = f"{args.output}/pileup_strid.npz"
        np.savez_compressed(path, **{key: final_df[key].values for key in keys})
        print(final_df)

    return None


if __name__ == "__main__":
    main()