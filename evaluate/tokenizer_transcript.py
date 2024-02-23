import numpy as np
import pandas as pd
import multiprocessing as mp
import os, argparse, tqdm, gc, glob
from utils.utils import oom_killer
from preprocess.tokenizer import create_segment_len_arr, sequence_to_kmer_token, expand_token_to_segment, \
    create_positional_token, create_move_token, segmented_signal_to_block, segmented_fft_to_block, create_target_mask


def parse_args():
    ## Usage: "python tokenizer.py --cpu {args.thread} --signal {signal_path} --output {args.output}"
    parser = argparse.ArgumentParser("Tokenize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu*0.9), help="Number of threads")
    parser.add_argument("--signal", "-s", type=str, required=True, help="Signal directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk", "-k", type=int, default=1000, help="Chunk size")
    args = parser.parse_args()
    if not os.path.exists(args.signal):
        raise FileNotFoundError(f"Signal directory {args.signal} does not exist")
    os.makedirs(args.output, exist_ok=True)
    return args


def tokenizer_worker(file_id_arr, signal_df_path_arr, output_dir, chunk,
                     kmer = 5, cb_len = 21, sampling = 5, sig_window = 5, fft_scale = 0.01):

    cb_lr_pad = (cb_len-kmer)//2
    trim = kmer//2

    for file_id, signal_df_path in tqdm.tqdm(zip(file_id_arr, signal_df_path_arr), total=len(file_id_arr)):
        oom_killer()
        signal_df = pd.read_pickle(f"{signal_df_path}")
        signal_df["block_id"] = signal_df["read_id"]
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
        signal_df = signal_df[["block_id", "motif", "kmer_token", "bq_token", "position_token", "signal_token",
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
        proc = mp.Process(target=tokenizer_worker, args=(file_id_arr[pid], signal_df_path_arr[pid], args.output, args.chunk))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    return None

if __name__ == "__main__":
    main()




