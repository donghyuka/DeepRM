import numpy as np
import pandas as pd
import multiprocessing as mp
import os, argparse, tqdm, gc, glob
from utils.utils import oom_killer

def sequence_to_kmer_token(seq, kmer):
    ## 1. change string to array of int - 0, 1, 2, 3
    seq = seq.upper()
    seq = seq.replace('A', '0')
    seq = seq.replace('C', '1')
    seq = seq.replace('G', '2')
    seq = seq.replace('T', '3')
    seq = seq.replace('U', '3')
    seq = np.array(list(seq), dtype=int)

    ## 2. convert to kmer token
    seq = [seq[i:kmer+i] for i in range(len(seq)-kmer+1)]
    seq = np.stack(seq, axis=1)
    quaternary = 4**np.arange(kmer).reshape(-1,1)
    seq = np.sum(seq * quaternary, axis=0) + 1 ## 0 is reserved for padding
    return seq


def create_segment_len_arr(segment_arr, sampling):
    segment_len_arr = np.array([len(x) for x in segment_arr], dtype=int)
    segment_len_arr = segment_len_arr // sampling
    return segment_len_arr

def expand_token_to_segment(token_arr, segment_len_arr):
    token = np.repeat(token_arr, segment_len_arr)
    return token


def create_positional_token(segment_len):
    ## create sinusoidal positional encoding
    position = np.arange(1, segment_len+1)
    return position


def create_move_token(segment_len_arr):
    token = np.arange(1, len(segment_len_arr)+1)
    token = np.repeat(token, segment_len_arr)
    return token


def segmented_signal_to_block(signal_segmented, segment_len_arr, kmer, sampling, sig_window):
    kmer_pad = (kmer-1)//2
    lr_pad = (sig_window-1)//2
    l_skip = np.sum(segment_len_arr[:kmer_pad])-lr_pad
    r_skip = np.sum(segment_len_arr[-kmer_pad:])-lr_pad
    assert l_skip >= 0, f"Left skip is negative: {l_skip}, segment_len_arr: {segment_len_arr}"
    assert r_skip >= 0, f"Right skip is negative: {r_skip}, segment_len_arr: {segment_len_arr}"
    signal_segmented = np.concatenate(signal_segmented)
    signal_segmented = np.stack(np.split(signal_segmented, len(signal_segmented) // sampling), axis=0)
    signal_segmented = np.array([np.concatenate(signal_segmented[i:i+sig_window]) for i in range(len(signal_segmented)-sig_window+1)])
    if r_skip > 0:
        signal_segmented = signal_segmented[l_skip:-r_skip]
    else:
        signal_segmented = signal_segmented[l_skip:]
    return signal_segmented


def segmented_fft_to_block(signal_segmented, fft_scale):
    signal_segmented = np.concatenate(signal_segmented) / fft_scale
    return signal_segmented

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
    args = parser.parse_args()
    if not os.path.exists(args.signal):
        raise FileNotFoundError(f"Signal directory {args.signal} does not exist")
    os.makedirs(args.output, exist_ok=True)
    return args


def tokenizer_worker(file_id_arr, signal_df_path_arr, output_dir, kmer = 5, cb_len = 21, sampling = 5, sig_window = 5,
                     fft_scale = 0.01, max_penalty = 10, chunk = 10000):

    cb_lr_pad = (cb_len-kmer)//2
    trim = kmer//2

    for file_id, signal_df_path in tqdm.tqdm(zip(file_id_arr, signal_df_path_arr), total=len(file_id_arr)):
        oom_killer()
        signal_df = pd.read_pickle(f"{signal_df_path}")
        signal_df = signal_df[signal_df["penalty"] <= max_penalty]
        signal_df["block_id"] = signal_df["read_id"] + "-" + signal_df["block_id"].astype(str)
        signal_df["block_score"] = signal_df["penalty"].apply(lambda x: 1-(x/max_penalty))
        signal_df["signal_fft"] = signal_df["signal_fft"].apply(lambda x: x[trim:-trim])
        signal_df["bq"] = signal_df["bq"].apply(lambda x: x[trim:-trim])
        signal_df["segment_len_arr"] = signal_df["signal_seg"].apply(lambda x: create_segment_len_arr(x, sampling))
        signal_df["signal_token"] = signal_df.apply(lambda x: segmented_signal_to_block(x["signal_seg"], x["segment_len_arr"],
                                                                                        kmer, sampling, sig_window), axis=1)
        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x[trim:-trim])
        signal_df["kmer_token"] = signal_df["motif"].apply(lambda x: sequence_to_kmer_token(x, kmer))
        signal_df["kmer_token"] = signal_df.apply(lambda x: expand_token_to_segment(x["kmer_token"], x["segment_len_arr"]), axis=1)
        signal_df["bq_token"] = signal_df.apply(lambda x: expand_token_to_segment(x["bq"], x["segment_len_arr"]), axis=1)
        signal_df["position_token"] = signal_df["segment_len_arr"].apply(lambda x: create_positional_token(np.sum(x)))
        signal_df["move_token"] = signal_df["segment_len_arr"].apply(lambda x: create_move_token(x))
        signal_df["spectrogram_token"] = signal_df["signal_fft"].apply(lambda x: segmented_fft_to_block(x, fft_scale))
        signal_df["target_mask"] = signal_df["segment_len_arr"].apply(lambda x: create_target_mask(x, cb_lr_pad))
        signal_df = signal_df[["block_id", "motif", "block_score", "kmer_token", "bq_token", "position_token", "signal_token",
                               "spectrogram_token", "move_token", "target_mask"]].copy()
        gc.collect()


        for chunk_idx in range(0, len(signal_df) // chunk + 1):
            chunk_df = signal_df.iloc[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len(signal_df))].copy()
            save_path = f"{output_dir}/tokenized_{file_id}-{chunk_idx}.pkl"
            chunk_df.to_pickle(save_path)

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
        proc = mp.Process(target=tokenizer_worker, args=(file_id_arr[pid], signal_df_path_arr[pid], args.output))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    return None

if __name__ == "__main__":
    main()




