## Train / Validation / Test split
## Keep Pos:Neg ratio of 1:1
## Keep uniform distribution of 256 5-mer motifs.
## Sample engineering dataset
##
## Output structure:
##  |---/main
##  |   |---/train
##  |   |   |---/pos
##  |   |   |---/neg
##  |   |---/val
##  |   |   |---/pos
##  |   |   |---/neg
##  |   |---/test
##  |       |---/pos
##  |       |---/neg
##  |
##  |---/engineering
##      |---/train
##      |   |---/pos
##      |   |---/neg
##      |---/val
##      |   |---/pos
##      |   |---/neg
##      |---/test
##          |---/pos
##          |---/neg
##



import numpy as np
import pandas as pd
import multiprocessing as mp
import os, argparse, tqdm, gc, glob

from utils.utils import printmessage


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--pos", dest="pos_path", type=str, required=True, nargs="+", help="Positive token files")
    args.add_argument("--neg", dest="neg_path", type=str, required=True, nargs="+", help="Negative token files")
    args.add_argument("--out", dest="out_path", type=str, required=True, help="Output directory")
    args.add_argument("--sam", dest="sampling", type=float, default=0.01, help="Sampling rate")
    args.add_argument("--max", dest="max_token_len", type=int, default=1000, help="Maximum token length")
    args.add_argument("--cpu", dest="cpu", type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args = args.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    return args

def get_motif_df_worker(df_path_list, return_list, kmer_size, cb_size):
    for df_path in df_path_list:
        df = pd.read_pickle(df_path)
        df["kmer"] = df["motif"].apply(lambda x: x[cb_size//2-kmer_size//2:cb_size//2+kmer_size//2+1])
        df["token_len"] = df["signal_token"].apply(lambda x: len(x))
        return_list.append(df[["block_id", "kmer", "token_len"]])
    return None


def get_motif_df(path_list, ncpu, max_token_len, kmer_size=5, cb_size=17):
    file_list = [y for x in path_list for y in glob.glob(f"{x}/*.pkl")]
    file_list = np.array_split(file_list, ncpu)
    manager = mp.Manager()
    return_list = manager.list()
    proc_list = []
    for files in file_list:
        proc = mp.Process(target=get_motif_df_worker, args=(files, return_list, kmer_size, cb_size))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    return_list = list(return_list)
    manager.shutdown()
    return_list = pd.concat(return_list)
    print(return_list["token_len"].describe())
    return_list = return_list[return_list["token_len"] <= max_token_len].reset_index(drop=True).copy()
    gc.collect()
    return return_list


def sample_and_save_df(id_set_list, out_path_list, in_path_list, ncpu, save_rows = 10000):
    in_file_list = [*glob.glob(f"{in_path_list}/*.pkl")]
    in_file_list = np.array_split(in_file_list, ncpu)
    proc_list = []
    for pid in range(ncpu):
        proc = mp.Process(target=sample_and_save_df_worker, args=(id_set_list, out_path_list, in_file_list[pid], save_rows))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    return None


def sample_and_save_df_worker(id_set_list, out_path_list, in_path_list, save_rows):
    for df_path in in_path_list:
        df = pd.read_pickle(df_path)
        for id_set, out_path in tqdm.tqdm(zip(id_set_list, out_path_list), total=len(id_set_list)):
            sample_df = df[df["block_id"].isin(id_set)]
            sample_df["token_len"] = sample_df["signal_token"].apply(lambda x: len(x))
            sample_df.sort_values("token_len", inplace=True)
            sample_df.reset_index(drop=True, inplace=True)
            for row_idx in range(0, len(sample_df), save_rows):
                save_df = sample_df[row_idx:min(row_idx+save_rows, len(sample_df))]
                save_df.to_pickle(out_path + df_path.split("/")[-1])
                gc.collect()
    return None
        

def split_dataset_kmer_balanced(kmer_df, split_ratio = [0.8, 0.1, 0.1], seed = 42):
    split_ratio = np.array(split_ratio) / np.sum(split_ratio)
    split_df_dict = {i:[] for i in range(len(split_ratio))}
    kmer_list = kmer_df["kmer"].unique()
    for kmer in kmer_list:
        kmer_sub_df = kmer_df[kmer_df["kmer"] == kmer]
        kmer_sub_df = kmer_sub_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        kmer_sub_df = np.array_split(kmer_sub_df, np.cumsum(split_ratio[:-1]*len(kmer_sub_df)).astype(int))
        for i in range(len(split_ratio)):
            split_df_dict[i].append(kmer_sub_df[i])
    for i in range(len(split_ratio)):
        split_df_dict[i] = pd.concat(split_df_dict[i])
    return split_df_dict


def sample_dataset_kmer_balanced(kmer_df, sample_ratio = 0.01, seed = 42):
    sample_list = []
    kmer_list = kmer_df["kmer"].unique()

    for kmer in kmer_list:
        kmer_sub_df = kmer_df[kmer_df["kmer"] == kmer]
        if sample_ratio > 1:
            kmer_sub_df = kmer_sub_df.sample(sample_ratio, random_state=seed)
        else:
            kmer_sub_df = kmer_sub_df.sample(frac=sample_ratio, random_state=seed)
        sample_list.append(kmer_sub_df)
    sample_df = pd.concat(sample_list)
    return sample_df


def main(seed = 42):
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    pos_cnt_kmer_df = get_motif_df(args.pos_path, args.cpu, args.max_token_len)
    neg_cnt_kmer_df = get_motif_df(args.neg_path, args.cpu, args.max_token_len)
    pos_cnt = len(pos_cnt_kmer_df)
    neg_cnt = len(neg_cnt_kmer_df)
    if pos_cnt > neg_cnt:
        pos_cnt_kmer_df = sample_dataset_kmer_balanced(pos_cnt_kmer_df, sample_ratio = neg_cnt , seed = seed)
    elif pos_cnt < neg_cnt:
        neg_cnt_kmer_df = sample_dataset_kmer_balanced(neg_cnt_kmer_df, sample_ratio = pos_cnt , seed = seed)
    else:
        pass
    pos_cnt_kmer_df_split = split_dataset_kmer_balanced(pos_cnt_kmer_df, seed = seed)
    neg_cnt_kmer_df_split = split_dataset_kmer_balanced(neg_cnt_kmer_df, seed = seed)

    pos_train = pos_cnt_kmer_df_split[0]
    pos_val = pos_cnt_kmer_df_split[1]
    pos_test = pos_cnt_kmer_df_split[2]
    neg_train = neg_cnt_kmer_df_split[0]
    neg_val = neg_cnt_kmer_df_split[1]
    neg_test = neg_cnt_kmer_df_split[2]
    
    pos_train_eng = sample_dataset_kmer_balanced(pos_train, sample_ratio = args.sampling, seed = seed)
    pos_val_eng = sample_dataset_kmer_balanced(pos_val, sample_ratio = args.sampling, seed = seed)
    pos_test_eng = sample_dataset_kmer_balanced(pos_test, sample_ratio = args.sampling, seed = seed)
    neg_train_eng = sample_dataset_kmer_balanced(neg_train, sample_ratio = args.sampling, seed = seed)
    neg_val_eng = sample_dataset_kmer_balanced(neg_val, sample_ratio = args.sampling, seed = seed)
    neg_test_eng = sample_dataset_kmer_balanced(neg_test, sample_ratio = args.sampling, seed = seed)

    pos_df_list = [pos_train, pos_val, pos_test, pos_train_eng, pos_val_eng, pos_test_eng]
    neg_df_list = [neg_train, neg_val, neg_test, neg_train_eng, neg_val_eng, neg_test_eng]

    pos_df_list = [set(df["block_id"]) for df in pos_df_list]
    neg_df_list = [set(df["block_id"]) for df in neg_df_list]

    pos_path_list = [f"{args.out_path}/main/train/pos/", f"{args.out_path}/main/val/pos/", f"{args.out_path}/main/test/pos/",
                        f"{args.out_path}/engineering/train/pos/", f"{args.out_path}/engineering/val/pos/", f"{args.out_path}/engineering/test/pos/"]
    neg_path_list = [f"{args.out_path}/main/train/neg/", f"{args.out_path}/main/val/neg/", f"{args.out_path}/main/test/neg/",
                        f"{args.out_path}/engineering/train/neg/", f"{args.out_path}/engineering/val/neg/", f"{args.out_path}/engineering/test/neg/"]

    sample_and_save_df(pos_df_list, pos_path_list, args.pos_path, args.cpu)
    sample_and_save_df(neg_df_list, neg_path_list, args.neg_path, args.cpu)

    return None



def main_pos(seed = 42):
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    printmessage("Loading tokenized data")
    pos_cnt_kmer_df = get_motif_df(args.pos_path, args.cpu)
    pos_cnt = len(pos_cnt_kmer_df)
    printmessage(f"Positive data points: {pos_cnt}")
    printmessage("Splitting dataset")
    pos_cnt_kmer_df_split = split_dataset_kmer_balanced(pos_cnt_kmer_df, seed = seed)

    pos_train = pos_cnt_kmer_df_split[0]
    pos_val = pos_cnt_kmer_df_split[1]
    pos_test = pos_cnt_kmer_df_split[2]

    printmessage("Sampling engineering dataset")
    pos_train_eng = sample_dataset_kmer_balanced(pos_train, sample_ratio = args.sampling, seed = seed)
    pos_val_eng = sample_dataset_kmer_balanced(pos_val, sample_ratio = args.sampling, seed = seed)
    pos_test_eng = sample_dataset_kmer_balanced(pos_test, sample_ratio = args.sampling, seed = seed)

    pos_df_list = [pos_train, pos_val, pos_test, pos_train_eng, pos_val_eng, pos_test_eng]
    pos_df_list = [set(df["block_id"]) for df in pos_df_list]
    pos_path_list = [f"{args.out_path}/main/train/pos/", f"{args.out_path}/main/val/pos/", f"{args.out_path}/main/test/pos/",
                        f"{args.out_path}/engineering/train/pos/", f"{args.out_path}/engineering/val/pos/", f"{args.out_path}/engineering/test/pos/"]

    printmessage("Saving dataset")
    sample_and_save_df(pos_df_list, pos_path_list, args.pos_path, args.cpu)

    return None


if __name__ == "__main__":
    main_pos()


