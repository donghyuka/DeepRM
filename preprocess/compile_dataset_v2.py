## Train / Validation / Test split
## Keep Pos:Neg ratio of 1:1
## Keep uniform distribution of 256 5-mer motifs.
## Sample engineering dataset
##
## Output structure:
##  |
##  |---/metadata
##  |
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
    args.add_argument("--max", dest="max_token_len", type=int, default=200, help="Maximum token length")
    args.add_argument("--cpu", dest="cpu", type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args.add_argument("--chk", dest="chunk", type=int, default=1000, help="Chunk size")
    args.add_argument("--seed", dest="seed", type=int, default=None, help="Random seed")
    args.add_argument("--score", dest="score", type=float, default=0.0, help="Block score threshold")
    args.add_argument("--ratio", dest="ratio", type=float, default=1, help="Negative-to-Positive ratio")
    args = args.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    return args


def get_metadata_df_worker(df_path_list, return_list, kmer_size, cb_size):
    df_list = []
    for df_path in df_path_list:
        df = pd.read_pickle(df_path)
        df["token_len"] = df["signal_token"].apply(lambda x: len(x))
        df["kmer"] = df["motif"].apply(lambda x: x[cb_size//2-kmer_size//2:cb_size//2+kmer_size//2+1])
        df = df[["block_id", "block_score", "kmer", "token_len"]].copy()
        # df["filename"] = df_path.split("/")[-1]
        df_list.append(df)
        del df
        gc.collect()
    return_list.append(pd.concat(df_list))
    del df_list
    gc.collect()
    return None


def get_metadata_df(path_list, ncpu, score_threshold, kmer_size=5, cb_size=21):
    file_list = [y for x in path_list for y in glob.glob(f"{x}/*.pkl")]
    file_list = np.array_split(file_list, ncpu)
    manager = mp.Manager()
    return_list = manager.list()
    proc_list = []
    for files in file_list:
        proc = mp.Process(target=get_metadata_df_worker, args=(files, return_list, kmer_size, cb_size))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()
    return_list = list(return_list)
    manager.shutdown()
    return_list = pd.concat(return_list)
    return_list = return_list[return_list["block_score"] >= score_threshold]
    return_list = return_list.reset_index(drop=True)
    print(return_list)
    gc.collect()
    return return_list


def sample_and_save_df(id_set_list, out_path_list, meta_in_path_list, meta_out_path_list, in_path_list,
                       ncpu, label, chunk, id_digit=6):

    in_file_list = [x for in_path in in_path_list for x in glob.glob(f"{in_path}/*.pkl")]
    in_file_list = np.array_split(in_file_list, ncpu)
    proc_list = []
    pid_digit = len(str(ncpu))
    man = mp.Manager()
    path_df_dict = man.dict()
    remainder_df_dict = man.dict()
    for path in out_path_list:
        path_df_dict[path] = man.list()
        remainder_df_dict[path] = man.list()

    for pid in range(ncpu):
        pid_str = str(pid).zfill(pid_digit)
        proc = mp.Process(target=sample_and_save_df_worker, args=(id_set_list, out_path_list, in_file_list[pid], pid_str,
                                                                  chunk, label, path_df_dict, remainder_df_dict, id_digit))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    pid_str = str(ncpu).zfill(pid_digit)
    out_df_id = 0

    for out_path, remainder_df_list in remainder_df_dict.items():
        sample_df = pd.concat(remainder_df_list)

        for chunk_idx in range(0, len(sample_df) // chunk + 1):
            out_df_id += 1
            chunk_df = sample_df.iloc[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len(sample_df))].copy()
            if len(chunk_df) == chunk:
                file_name = f"{pid_str}-{str(out_df_id).zfill(id_digit)}.pkl"
                save_path = f"{out_path}/{file_name}"
                chunk_df.to_pickle(save_path)
                path_df = pd.DataFrame({"file_name": file_name, "block_id": chunk_df["block_id"]})
                path_df_dict[out_path].append(path_df)

    del remainder_df_dict
    gc.collect()

    for out_path, path_df_list in path_df_dict.items():
        path_df = pd.concat(path_df_list)
        meta_df = pd.read_pickle(meta_in_path_list[out_path_list.index(out_path)])
        meta_df = meta_df.merge(path_df, on="block_id", how="inner")
        meta_df.to_pickle(meta_out_path_list[out_path_list.index(out_path)])

    del path_df_dict
    gc.collect()

    return None


def sample_and_save_df_worker(id_set_list, out_path_list, in_path_list, pid_str, chunk, label,
                              path_df_dict_shared, remainder_df_dict_shared, id_zfill):

    out_df_id = 0
    buffer_dict = {out_path:None for out_path in out_path_list}
    path_df_dict = {out_path:[] for out_path in out_path_list}

    for df_path in tqdm.tqdm(in_path_list):
        df = pd.read_pickle(df_path)
        df["label"] = label
        df["token_len"] = df["signal_token"].apply(lambda x: len(x))

        for id_set, out_path in zip(id_set_list, out_path_list):
            sample_df = df[df["block_id"].isin(id_set)].copy()

            if buffer_dict[out_path] is not None:
                sample_df = pd.concat([buffer_dict[out_path], sample_df])
                buffer_dict[out_path] = None
                gc.collect()

            sample_df.reset_index(drop=True, inplace=True)

            for chunk_idx in range(0, len(sample_df) // chunk + 1):
                out_df_id += 1
                chunk_df = sample_df.iloc[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len(sample_df))].copy()
                if len(chunk_df) == chunk:
                    file_name = f"{pid_str}-{str(out_df_id).zfill(id_zfill)}.pkl"
                    save_path = f"{out_path}/{file_name}"
                    chunk_df.to_pickle(save_path)
                    path_df = pd.DataFrame({"file_name": file_name, "block_id": chunk_df["block_id"]})
                    path_df_dict[out_path].append(path_df)
                    del chunk_df
                    gc.collect()

                else:
                    ## should happen only once per iteration
                    buffer_dict[out_path] = chunk_df

            del sample_df
            gc.collect()


    for out_path, buffer in buffer_dict.items():
        if buffer is not None:
            remainder_df_dict_shared[out_path].append(buffer)
    del buffer_dict
    gc.collect()

    for out_path, path_df_list in path_df_dict.items():
        if len(path_df_list) > 0:
            path_df_dict_shared[out_path].append(pd.concat(path_df_list))
    del path_df_dict
    gc.collect()

    return None
        

def split_dataset_kmer_balanced(kmer_df, split_ratio = [0.8, 0.1, 0.1], seed = 42):
    split_ratio = np.array(split_ratio) / np.sum(split_ratio)
    split_df_dict = {i:[] for i in range(len(split_ratio))}
    kmer_list = kmer_df["kmer"].unique()
    for kmer in tqdm.tqdm(kmer_list):
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

    if sample_ratio > 1:
        kmer_count_dict = kmer_df["kmer"].value_counts().to_dict()
        kmer_list = list(kmer_count_dict.keys())
        kmer_sample_count_arr = np.array([kmer_count_dict[kmer] for kmer in kmer_list]) * sample_ratio / len(kmer_df)
        kmer_sample_count_arr = np.floor(kmer_sample_count_arr).astype(int)
        sample_sum = np.sum(kmer_sample_count_arr)
        sample_diff = sample_ratio - sample_sum
        choice_idx = np.random.choice(len(kmer_list), size=sample_diff, replace=False)
        for idx in choice_idx:
            kmer_sample_count_arr[idx] += 1
        for kmer, sample_count in zip(kmer_list, kmer_sample_count_arr):
            kmer_sub_df = kmer_df[kmer_df["kmer"] == kmer]
            kmer_sub_df = kmer_sub_df.sample(n=sample_count, replace=True, random_state=seed)
            sample_list.append(kmer_sub_df)

    else:
        kmer_list = kmer_df["kmer"].unique()
        for kmer in kmer_list:
            kmer_sub_df = kmer_df[kmer_df["kmer"] == kmer]
            kmer_sub_df = kmer_sub_df.sample(frac=sample_ratio, random_state=seed)
            sample_list.append(kmer_sub_df)

    sample_df = pd.concat(sample_list, ignore_index=True)

    return sample_df


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(0, 1000000)

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(f"{args.out_path}/metadata", exist_ok=True)
    os.makedirs(f"{args.out_path}/metadata_path", exist_ok=True)

    pos_metadata_df = get_metadata_df(args.pos_path, args.cpu, args.score)
    neg_metadata_df = get_metadata_df(args.neg_path, args.cpu, args.score)
    pos_cnt = len(pos_metadata_df)
    neg_cnt = len(neg_metadata_df)

    pos_metadata_df["label"] = 1
    neg_metadata_df["label"] = 0
    
    pos_metadata_df.to_pickle(f"{args.out_path}/metadata/all_pos.pkl")
    neg_metadata_df.to_pickle(f"{args.out_path}/metadata/all_neg.pkl")

    pos_metadata_df = pos_metadata_df[pos_metadata_df["token_len"] <= args.max_token_len].reset_index(drop=True).copy()
    neg_metadata_df = neg_metadata_df[neg_metadata_df["token_len"] <= args.max_token_len].reset_index(drop=True).copy()

    printmessage("Positive before sampling: ", pos_cnt)
    printmessage("Negative before sampling: ", neg_cnt)

    assert neg_cnt > pos_cnt, "Negative count should be greater than positive count."

    if pos_cnt * args.ratio > neg_cnt:
        pos_cnt = neg_cnt // args.ratio
        pos_metadata_df = sample_dataset_kmer_balanced(pos_metadata_df, sample_ratio = pos_cnt , seed = args.seed)

    else:
        neg_cnt = int(pos_cnt * args.ratio)
        neg_metadata_df = sample_dataset_kmer_balanced(neg_metadata_df, sample_ratio = neg_cnt , seed = args.seed)

    printmessage("Positive after sampling: ", len(pos_metadata_df))
    printmessage("Negative after sampling: ", len(neg_metadata_df))

    pos_metadata_df_split = split_dataset_kmer_balanced(pos_metadata_df, seed = args.seed)
    neg_metadata_df_split = split_dataset_kmer_balanced(neg_metadata_df, seed = args.seed)

    pos_train = pos_metadata_df_split[0]
    pos_val = pos_metadata_df_split[1]
    pos_test = pos_metadata_df_split[2]
    neg_train = neg_metadata_df_split[0]
    neg_val = neg_metadata_df_split[1]
    neg_test = neg_metadata_df_split[2]
    
    pos_train_eng = sample_dataset_kmer_balanced(pos_train, sample_ratio = args.sampling, seed = args.seed)
    pos_val_eng = sample_dataset_kmer_balanced(pos_val, sample_ratio = args.sampling, seed = args.seed)
    pos_test_eng = sample_dataset_kmer_balanced(pos_test, sample_ratio = args.sampling, seed = args.seed)
    neg_train_eng = sample_dataset_kmer_balanced(neg_train, sample_ratio = args.sampling, seed = args.seed)
    neg_val_eng = sample_dataset_kmer_balanced(neg_val, sample_ratio = args.sampling, seed = args.seed)
    neg_test_eng = sample_dataset_kmer_balanced(neg_test, sample_ratio = args.sampling, seed = args.seed)

    pos_df_list = [pos_train, pos_val, pos_test, pos_train_eng, pos_val_eng, pos_test_eng]
    neg_df_list = [neg_train, neg_val, neg_test, neg_train_eng, neg_val_eng, neg_test_eng]

    pos_train.to_pickle(f"{args.out_path}/metadata/pos_train.pkl")
    pos_val.to_pickle(f"{args.out_path}/metadata/pos_val.pkl")
    pos_test.to_pickle(f"{args.out_path}/metadata/pos_test.pkl")
    pos_train_eng.to_pickle(f"{args.out_path}/metadata/pos_train_eng.pkl")
    pos_val_eng.to_pickle(f"{args.out_path}/metadata/pos_val_eng.pkl")
    pos_test_eng.to_pickle(f"{args.out_path}/metadata/pos_test_eng.pkl")
    neg_train.to_pickle(f"{args.out_path}/metadata/neg_train.pkl")
    neg_val.to_pickle(f"{args.out_path}/metadata/neg_val.pkl")
    neg_test.to_pickle(f"{args.out_path}/metadata/neg_test.pkl")
    neg_train_eng.to_pickle(f"{args.out_path}/metadata/neg_train_eng.pkl")
    neg_val_eng.to_pickle(f"{args.out_path}/metadata/neg_val_eng.pkl")
    neg_test_eng.to_pickle(f"{args.out_path}/metadata/neg_test_eng.pkl")

    pos_df_list = [set(df["block_id"]) for df in pos_df_list]
    neg_df_list = [set(df["block_id"]) for df in neg_df_list]

    pos_path_list = [f"{args.out_path}/main/train/pos/", f"{args.out_path}/main/val/pos/", f"{args.out_path}/main/test/pos/",
                        f"{args.out_path}/engineering/train/pos/", f"{args.out_path}/engineering/val/pos/", f"{args.out_path}/engineering/test/pos/"]
    neg_path_list = [f"{args.out_path}/main/train/neg/", f"{args.out_path}/main/val/neg/", f"{args.out_path}/main/test/neg/",
                        f"{args.out_path}/engineering/train/neg/", f"{args.out_path}/engineering/val/neg/", f"{args.out_path}/engineering/test/neg/"]
    pos_metadata_path_list = [f"{args.out_path}/metadata_path/pos_train.pkl", f"{args.out_path}/metadata_path/pos_val.pkl", f"{args.out_path}/metadata_path/pos_test.pkl",
                        f"{args.out_path}/metadata_path/pos_train_eng.pkl", f"{args.out_path}/metadata_path/pos_val_eng.pkl", f"{args.out_path}/metadata_path/pos_test_eng.pkl"]
    neg_metadata_path_list = [f"{args.out_path}/metadata_path/neg_train.pkl", f"{args.out_path}/metadata_path/neg_val.pkl", f"{args.out_path}/metadata_path/neg_test.pkl",
                        f"{args.out_path}/metadata_path/neg_train_eng.pkl", f"{args.out_path}/metadata_path/neg_val_eng.pkl", f"{args.out_path}/metadata_path/neg_test_eng.pkl"]
    pos_original_metadata_path_list = [f"{args.out_path}/metadata/pos_train.pkl", f"{args.out_path}/metadata/pos_val.pkl", f"{args.out_path}/metadata/pos_test.pkl",
                        f"{args.out_path}/metadata/pos_train_eng.pkl", f"{args.out_path}/metadata/pos_val_eng.pkl", f"{args.out_path}/metadata/pos_test_eng.pkl"]
    neg_original_metadata_path_list = [f"{args.out_path}/metadata/neg_train.pkl", f"{args.out_path}/metadata/neg_val.pkl", f"{args.out_path}/metadata/neg_test.pkl",
                        f"{args.out_path}/metadata/neg_train_eng.pkl", f"{args.out_path}/metadata/neg_val_eng.pkl", f"{args.out_path}/metadata/neg_test_eng.pkl"]

    for path in pos_path_list:
        os.makedirs(path, exist_ok=True)
    for path in neg_path_list:
        os.makedirs(path, exist_ok=True)
    sample_and_save_df(pos_df_list, pos_path_list, pos_original_metadata_path_list, pos_metadata_path_list, args.pos_path, args.cpu, label = 0, chunk = args.chunk)
    sample_and_save_df(neg_df_list, neg_path_list, neg_original_metadata_path_list, neg_metadata_path_list, args.neg_path, args.cpu, label = 1, chunk = args.chunk)

    return None


def main2():
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(0, 1000000)

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(f"{args.out_path}/metadata", exist_ok=True)
    os.makedirs(f"{args.out_path}/metadata_path", exist_ok=True)

    pos_train = pd.read_pickle(f"{args.out_path}/metadata/pos_train.pkl")
    pos_val = pd.read_pickle(f"{args.out_path}/metadata/pos_val.pkl")
    pos_test = pd.read_pickle(f"{args.out_path}/metadata/pos_test.pkl")
    neg_train = pd.read_pickle(f"{args.out_path}/metadata/neg_train.pkl")
    neg_val = pd.read_pickle(f"{args.out_path}/metadata/neg_val.pkl")
    neg_test = pd.read_pickle(f"{args.out_path}/metadata/neg_test.pkl")

    pos_train_count = int(len(pos_train) * args.sampling)
    neg_train_count = int(pos_train_count * args.ratio)
    pos_val_count = int(len(pos_val) * args.sampling)
    neg_val_count = int(pos_val_count * args.ratio)
    pos_test_count = int(len(pos_test) * args.sampling)
    neg_test_count = int(pos_test_count * args.ratio)

    print(pos_train_count, neg_train_count, pos_val_count, neg_val_count, pos_test_count, neg_test_count)


    pos_train_eng = sample_dataset_kmer_balanced(pos_train, sample_ratio = pos_train_count, seed = args.seed)
    pos_val_eng = sample_dataset_kmer_balanced(pos_val, sample_ratio = pos_val_count, seed = args.seed)
    pos_test_eng = sample_dataset_kmer_balanced(pos_test, sample_ratio = pos_test_count, seed = args.seed)
    neg_train_eng = sample_dataset_kmer_balanced(neg_train, sample_ratio = neg_train_count, seed = args.seed)
    neg_val_eng = sample_dataset_kmer_balanced(neg_val, sample_ratio = neg_val_count, seed = args.seed)
    neg_test_eng = sample_dataset_kmer_balanced(neg_test, sample_ratio = neg_test_count, seed = args.seed)

    pos_df_list = [pos_train_eng, pos_val_eng, pos_test_eng]
    neg_df_list = [neg_train_eng, neg_val_eng, neg_test_eng]

    pos_train_eng.to_pickle(f"{args.out_path}/metadata/pos_train_imbx10.pkl")
    pos_val_eng.to_pickle(f"{args.out_path}/metadata/pos_val_imbx10.pkl")
    pos_test_eng.to_pickle(f"{args.out_path}/metadata/pos_test_imbx10.pkl")
    neg_train_eng.to_pickle(f"{args.out_path}/metadata/neg_train_imbx10.pkl")
    neg_val_eng.to_pickle(f"{args.out_path}/metadata/neg_val_imbx10.pkl")
    neg_test_eng.to_pickle(f"{args.out_path}/metadata/neg_test_imbx10.pkl")

    pos_df_list = [set(df["block_id"]) for df in pos_df_list]
    neg_df_list = [set(df["block_id"]) for df in neg_df_list]

    pos_path_list = [f"{args.out_path}/imbx10/train/pos/", f"{args.out_path}/imbx10/val/pos/", f"{args.out_path}/imbx10/test/pos/"]
    neg_path_list = [f"{args.out_path}/imbx10/train/neg/", f"{args.out_path}/imbx10/val/neg/", f"{args.out_path}/imbx10/test/neg/"]
    pos_metadata_path_list = [
                        f"{args.out_path}/metadata_path/pos_train_imbx10.pkl", f"{args.out_path}/metadata_path/pos_val_imbx10.pkl", f"{args.out_path}/metadata_path/pos_test_imbx10.pkl"]
    neg_metadata_path_list = [
                        f"{args.out_path}/metadata_path/neg_train_imbx10.pkl", f"{args.out_path}/metadata_path/neg_val_imbx10.pkl", f"{args.out_path}/metadata_path/neg_test_imbx10.pkl"]
    pos_original_metadata_path_list = [
                        f"{args.out_path}/metadata/pos_train_imbx10.pkl", f"{args.out_path}/metadata/pos_val_imbx10.pkl", f"{args.out_path}/metadata/pos_test_imbx10.pkl"]
    neg_original_metadata_path_list = [
                        f"{args.out_path}/metadata/neg_train_imbx10.pkl", f"{args.out_path}/metadata/neg_val_imbx10.pkl", f"{args.out_path}/metadata/neg_test_imbx10.pkl"]

    for path in pos_path_list:
        os.makedirs(path, exist_ok=True)
    for path in neg_path_list:
        os.makedirs(path, exist_ok=True)
    sample_and_save_df(pos_df_list, pos_path_list, pos_original_metadata_path_list, pos_metadata_path_list, args.pos_path, args.cpu, label = 0, chunk = args.chunk)
    sample_and_save_df(neg_df_list, neg_path_list, neg_original_metadata_path_list, neg_metadata_path_list, args.neg_path, args.cpu, label = 1, chunk = args.chunk)

    return None


if __name__ == "__main__":
    main()


