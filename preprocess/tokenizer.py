import numpy as np
import pandas as pd
import multiprocessing as mp
import os, argparse, tqdm, gc, glob

def sequence_to_kmer_token(seq, kmer = 5):
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
    seq = np.sum(seq * quaternary, axis=0)
    return seq


def create_segment_len_arr(segment_arr, sampling = 5):
    segment_len_arr = np.array([len(x) for x in segment_arr], dtype=int)
    segment_len_arr = segment_len_arr // sampling
    return segment_len_arr

def expand_token_to_segment(token_arr, segment_len_arr):
    token = np.repeat(token_arr, segment_len_arr)
    return token


def create_positional_encoding(segment_len_arr):
    len_base = len(segment_len_arr) - 1
    pe = np.arange(0,1+1/len_base,1/len_base)
    pe = np.repeat(pe, segment_len_arr)
    return pe


def segmented_signal_to_block(signal_segmented, sampling = 5, window = 5):
    # Add windowing (overlapping signal patch)
    signal_segmented = np.concatenate(signal_segmented)
    signal_segmented = np.stack(np.split(signal_segmented, len(signal_segmented) // sampling), axis=0)
    return signal_segmented


def segmented_fft_to_block(signal_segmented):
    signal_segmented = np.concatenate(signal_segmented)
    return signal_segmented

def create_target_mask(segment_len_arr, lr_pad = 8):
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
    parser.add_argument("--block", "-b", type=str, required=True, help="Block dataframe")
    args = parser.parse_args()
    if not os.path.exists(args.signal):
        raise FileNotFoundError(f"Signal directory {args.signal} does not exist")
    if not os.path.exists(args.block):
        raise FileNotFoundError(f"Block dataframe {args.block} does not exist")
    os.makedirs(args.output, exist_ok=True)
    return args


def tokenizer_worker(file_id_arr, signal_df_path_arr, output_dir, kmer = 5, cb_len = 21):
    lr_pad = (cb_len-kmer)//2
    trim = kmer//2

    for file_id, signal_df_path in tqdm.tqdm(zip(file_id_arr, signal_df_path_arr), total=len(file_id_arr)):
        signal_df = pd.read_pickle(f"{signal_df_path}")

        print("PASS 1")
        # signal_df["signal_seg"] = signal_df["signal_seg"].apply(lambda x: x[trim:-trim])
        signal_df["signal_fft"] = signal_df["signal_fft"].apply(lambda x: x[trim:-trim])
        # signal_df["bq"] = signal_df["bq"].apply(lambda x: x[trim:-trim])
        print("PASS 2")

        signal_df["segment_len_arr"] = signal_df["signal_seg"].apply(lambda x: create_segment_len_arr(x))
        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x[trim:-trim])
        signal_df["kmer_token"] = signal_df["motif"].apply(lambda x: sequence_to_kmer_token(x, kmer))
        signal_df["kmer_token"] = signal_df.apply(lambda x: expand_token_to_segment(x["kmer_token"], x["segment_len_arr"]), axis=1)
        # signal_df["bq_token"] = signal_df["bq"].apply(lambda x: expand_token_to_segment(x, x["segment_len_arr"]))

        print("PASS 3")
        signal_df["position_token"] = signal_df["segment_len_arr"].apply(lambda x: create_positional_encoding(x))
        signal_df["signal_token"] = signal_df["signal_seg"].apply(lambda x: segmented_signal_to_block(x))

        print("PASS 4")
        signal_df["spectrogram_token"] = signal_df["signal_fft"].apply(lambda x: segmented_fft_to_block(x))
        signal_df["target_mask"] = signal_df["segment_len_arr"].apply(lambda x: create_target_mask(x, lr_pad))

        print("PASS 5")
        signal_df = signal_df[["id", "kmer_token", "position_token", "signal_token", "spectrogram_token", "target_mask"]]
        # signal_df = signal_df[["id", "kmer_token", "bq_token", "position_token", "signal_token", "spectrogram_token", "target_mask"]]
        signal_df["kmer_len"] = signal_df["kmer_token"].apply(lambda x: len(x))
        signal_df["spec_len"] = signal_df["spectrogram_token"].apply(lambda x: len(x))
        signal_df["sig_len"] = signal_df["signal_token"].apply(lambda x: len(x))
        signal_df["pos_len"] = signal_df["position_token"].apply(lambda x: len(x))
        signal_df["mask_len"] = signal_df["target_mask"].apply(lambda x: len(x))

        signal_df.to_pickle(f"{output_dir}/tokenized_{file_id}.pkl")
        print(signal_df[['kmer_len', 'spec_len', 'sig_len', 'pos_len', 'mask_len']])
        del signal_df
        gc.collect()

    return None


def main():
    args = parse_args()
    signal_df_path_arr = glob.glob(f"{args.signal}/*.pkl")
    signal_df_path_arr = np.array_split(signal_df_path_arr, args.cpu)
    file_id_arr = np.array_split(np.arange(len(signal_df_path_arr)), args.cpu)

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




