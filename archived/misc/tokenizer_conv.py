import numpy as np
import pandas as pd
import multiprocessing as mp
import os, argparse, tqdm, gc, glob
from utils.utils import oom_killer, seq_to_onehot

def seq_to_onehot_kmer(seq, kmer=5):
    onehot = seq_to_onehot(seq)
    stack_list = []
    for i in range(kmer-1):
        stack_list.append(onehot[i:-(kmer-i-1)])
    stack_list.append(onehot[kmer-1:])
    stack_arr = np.stack(stack_list, axis=1)
    return stack_arr


def create_segment_len_arr(segment_arr):
    segment_len_arr = np.array([len(x) for x in segment_arr], dtype=int)
    return segment_len_arr


def expand_token_to_segment(token_arr, segment_len_arr):
    token = np.repeat(token_arr, segment_len_arr, axis=0)
    return token


def create_move_token(segment_len_arr):
    token = np.arange(1, len(segment_len_arr)+1)
    token = np.repeat(token, segment_len_arr)
    return token


def create_target_mask(segment_len_arr, lr_pad):
    binary_mask = np.zeros(2*lr_pad+1)
    binary_mask[lr_pad] = 1
    binary_mask = np.repeat(binary_mask, segment_len_arr)
    return binary_mask


def parse_args():
    ## Usage: "python tokenizer.py --cpu {args.thread} --signal {signal_path} --output {args.output}"
    parser = argparse.ArgumentParser("Tokenize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu*0.9), help="Number of threads")
    parser.add_argument("--signal", "-s", type=str, required=True, help="Signal directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--qcut", "-q", type=str, default=0.9, help = "Block Quality Cutoff")
    args = parser.parse_args()
    if not os.path.exists(args.signal):
        raise FileNotFoundError(f"Signal directory {args.signal} does not exist")
    os.makedirs(args.output, exist_ok=True)
    return args


def tokenizer_worker(file_id_arr, signal_df_path_arr, output_dir, qcut, max_penalty = 10, chunk = 10000, cb_len = 21, kmer = 5):
    trim = kmer//2
    cb_lr_pad = (cb_len-kmer)//2

    chunk_buffer = []

    for file_id, signal_df_path in tqdm.tqdm(zip(file_id_arr, signal_df_path_arr), total=len(file_id_arr)):
        oom_killer()
        signal_df = pd.read_pickle(f"{signal_df_path}")
        signal_df = signal_df[signal_df["penalty"] <= max_penalty]
        signal_df["block_score"] = signal_df["penalty"].apply(lambda x: 1-(x/max_penalty))
        signal_df = signal_df[signal_df["block_score"] >= qcut].copy()
        gc.collect()

        signal_df["block_id"] = signal_df["read_id"] + "-" + signal_df["block_id"].astype(str)
        signal_df["segment_len_arr"] = signal_df["signal_seg"].apply(lambda x: create_segment_len_arr(x))
        try:
            signal_df["signal_seg"] = signal_df["signal_seg"].apply(np.concatenate)
        except:
            print(f"Error in {file_id}")
            continue

        signal_df.rename(columns={"signal_seg": "signal_token", "bq": "bq_token"}, inplace=True)
        signal_df = signal_df[signal_df["signal_token"].notnull()].copy()

        if len(signal_df) == 0:
            continue

        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x[trim:-trim])
        signal_df["signal_trim_front"] = signal_df["segment_len_arr"].apply(lambda x: np.sum(x[:trim]))
        signal_df["signal_trim_back"] = signal_df["segment_len_arr"].apply(lambda x: np.sum(x[-trim:]))
        signal_df["signal_token"] = signal_df.apply(lambda x: x["signal_token"][x["signal_trim_front"]:-x["signal_trim_back"]], axis=1)
        signal_df["seq_token"] = signal_df["motif"].apply(lambda x: seq_to_onehot_kmer(x,kmer))
        signal_df["seq_token"] = signal_df.apply(lambda x: expand_token_to_segment(x["seq_token"], x["segment_len_arr"]), axis=1)
        signal_df["bq_token"] = signal_df["bq_token"].apply(lambda x: x[trim:-trim])
        signal_df["bq_token"] = signal_df.apply(lambda x: expand_token_to_segment(x["bq_token"], x["segment_len_arr"]), axis=1)
        signal_df["move_token"] = signal_df["segment_len_arr"].apply(lambda x: create_move_token(x))
        signal_df["target_mask"] = signal_df["segment_len_arr"].apply(lambda x: create_target_mask(x, cb_lr_pad))
        signal_df = signal_df[["block_id", "seq_token", "bq_token", "signal_token", "move_token", "target_mask"]].copy()
        gc.collect()


        if len(chunk_buffer) > 0:
            signal_df = pd.concat([*chunk_buffer, signal_df])
            chunk_buffer = []

        for chunk_idx in range(0, len(signal_df) // chunk + 1):
            chunk_df = signal_df.iloc[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len(signal_df))].copy()
            if len(chunk_df) >= chunk:
                save_path = f"{output_dir}/tokenized_{file_id}-{chunk_idx}.pkl"
                chunk_df.to_pickle(save_path)
            else:
                chunk_buffer.append(chunk_df)

        del signal_df, chunk_df

        gc.collect()

    return None


def main():
    args = parse_args()
    signal_df_path_arr = glob.glob(f"{args.signal}/*.pkl")
    file_count = len(signal_df_path_arr)
    signal_df_path_arr = np.array_split(signal_df_path_arr, args.cpu)
    file_id_arr = np.array_split(np.arange(file_count), args.cpu)
    proc_list = []
    for pid in range(args.cpu):
        proc = mp.Process(target=tokenizer_worker, args=(file_id_arr[pid], signal_df_path_arr[pid], args.output, args.qcut))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    return None

if __name__ == "__main__":
    main()




