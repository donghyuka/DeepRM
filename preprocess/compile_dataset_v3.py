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

from utils.utils import printmessage, oom_killer


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--pos", dest="pos_path", type=str, default=None, nargs="+", help="Positive token files")
    args.add_argument("--neg", dest="neg_path", type=str, default=None, nargs="+", help="Negative token files")
    args.add_argument("--out", dest="out_path", type=str, required=True, help="Output directory")
    args.add_argument("--max", dest="max_token_len", type=int, default=200, help="Maximum token length")
    args.add_argument("--cpu", dest="cpu", type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args.add_argument("--chk", dest="chunk", type=int, default=1000, help="Chunk size")
    args.add_argument("--seed", dest="seed", type=int, default=None, help="Random seed")
    args.add_argument("--score", dest="score", type=float, nargs="+", default=[0.0], help="Block score threshold")
    args = args.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    return args



def sample_and_save_df(in_path_list, out_path, ncpu, label, chunk, min_score_list, max_token_len,
                       id_digit=8, shuffle = True):

    in_file_list = [x for in_path in in_path_list for x in glob.glob(f"{in_path}/*.pkl")]
    if shuffle:
        in_file_list = np.random.permutation(in_file_list)
    in_file_list = np.array_split(in_file_list, ncpu)
    proc_list = []
    pid_digit = len(str(ncpu))
    man = mp.Manager()
    remainder_df_dict = man.dict()
    label_str = {0:"neg", 1:"pos"}[label]
    set_split_dict = {"train":0.9, "val":0.1}

    for set_name in set_split_dict:
        for score in min_score_list:
            remainder_df_dict[(set_name,score)] = man.list()

    for pid in range(ncpu):
        pid_str = str(pid).zfill(pid_digit)
        proc = mp.Process(target=sample_and_save_df_worker, args=(in_file_list[pid], out_path, label_str,
                                                                  set_split_dict, pid_str, chunk, label, min_score_list,
                                                                  max_token_len, remainder_df_dict, id_digit, shuffle))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    pid_str = str(ncpu).zfill(pid_digit)
    out_df_id = 0

    for key in remainder_df_dict:
        set_name, min_score = key
        remainder_df_list = remainder_df_dict[key]
        if len(remainder_df_list) > 0:
            remainder_df = pd.concat(remainder_df_list).reset_index(drop=True)
            for chunk_idx in range(0, len(remainder_df) // chunk + 1):
                out_df_id += 1
                chunk_df = remainder_df.iloc[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len(remainder_df))].copy().reset_index(drop=True)
                if len(chunk_df) == chunk:
                    file_name = f"{pid_str}-{str(out_df_id).zfill(id_digit)}.pkl"
                    save_path = f"{out_path}/score-{min_score}/{set_name}/{label_str}/{file_name}"
                    chunk_df.to_pickle(save_path)
                    del chunk_df
                    gc.collect()
            del remainder_df
            gc.collect()
    del remainder_df_dict
    gc.collect()

    return None


def sample_and_save_df_worker(in_file_list, out_path, label_str, set_split_dict, pid_str,
                              chunk, label, min_score_list, max_token_len, remainder_df_dict, id_digit, shuffle):

    out_df_id = 0
    buffer_dict = {key:None for key in remainder_df_dict.keys()}

    for df_path in tqdm.tqdm(in_file_list):
        oom_killer()
        orig_df = pd.read_pickle(df_path)
        orig_df["label"] = label
        orig_df["token_len"] = orig_df["signal_token"].apply(lambda x: len(x))
        
        orig_df = orig_df[orig_df["token_len"] <= max_token_len]

        for min_score in min_score_list:

            score_df = orig_df[orig_df["block_score"] >= min_score].copy()

            if shuffle:
                score_df = score_df.sample(frac=1).reset_index(drop=True)
            else:
                score_df = score_df.reset_index(drop=True)

            ## slice for set by inex - use cumsum to get the index
            set_idx = np.cumsum([0] + [int(np.floor(len(score_df) * split_ratio)) for split_ratio in set_split_dict.values()])

            for idx, set_name in enumerate(set_split_dict):
                set_idx_start = set_idx[idx]
                set_idx_end = set_idx[idx+1]
                set_df = score_df.iloc[set_idx_start:set_idx_end].reset_index(drop=True)

                buffer = buffer_dict[(set_name,min_score)]
                if buffer is not None:
                    set_df = pd.concat([buffer, set_df]).reset_index(drop=True)
                    buffer_dict[(set_name,min_score)] = None
                    gc.collect()

                for chunk_idx in range(0, len(set_df) // chunk + 1):
                    out_df_id += 1
                    chunk_df = set_df.iloc[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len(set_df))].copy().reset_index(drop=True)
                    if len(chunk_df) == chunk:
                        file_name = f"{pid_str}-{str(out_df_id).zfill(id_digit)}.pkl"
                        save_path = f"{out_path}/score-{min_score}/{set_name}/{label_str}/{file_name}"
                        chunk_df.to_pickle(save_path)
                        del chunk_df
                        gc.collect()

                    else:
                        ## should happen only once per iteration
                        buffer_dict[(set_name,min_score)] = chunk_df

                del set_df
                gc.collect()
            del score_df
            gc.collect()
        del orig_df
        gc.collect()

    for key in buffer_dict:
        if buffer_dict[key] is not None:
            remainder_df_dict[key].append(buffer_dict[key])
    del buffer_dict
    gc.collect()

    return None


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(0, 1000000)

    os.makedirs(args.out_path, exist_ok=True)

    for set_name in ["train", "val"]:
        for score in args.score:
            for label in ["pos", "neg"]:
                os.makedirs(f"{args.out_path}/score-{score}/{set_name}/{label}", exist_ok=True)
    if args.pos_path is not None:
        sample_and_save_df(args.pos_path, args.out_path, args.cpu, label = 1,
                           chunk = args.chunk, min_score_list = args.score, max_token_len = args.max_token_len)
    if args.neg_path is not None:
        sample_and_save_df(args.neg_path, args.out_path, args.cpu, label = 0,
                           chunk = args.chunk, min_score_list = args.score, max_token_len = args.max_token_len)

    return None


if __name__ == "__main__":
    main()


