import argparse, gc, os, sys, pickle, re, pod5, pysam, tqdm, glob, toml
import multiprocessing as mp
import numpy as np
import pandas as pd
from utils.utils import mean_phred, oom_killer, printmessage



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
    seq = seq.astype(np.int16)
    return seq


def create_segment_len_arr(segment_arr, sampling):
    segment_len_arr = np.array([len(x) for x in segment_arr], dtype=int)
    segment_len_arr = segment_len_arr // sampling
    return segment_len_arr


def expand_token_to_segment(token_arr, segment_len_arr):
    token = np.repeat(token_arr, segment_len_arr)
    return token


def create_move_token(segment_len_arr):
    token = np.arange(1, len(segment_len_arr)+1, dtype=np.uint8)
    token = np.repeat(token, segment_len_arr)
    return token


def create_target_mask(segment_len_arr, lr_pad):
    binary_mask = np.zeros(2*lr_pad+1, dtype=np.uint8)
    binary_mask[lr_pad] = 1
    binary_mask = np.repeat(binary_mask, segment_len_arr)
    return binary_mask


def trim_scale_segment_signal(signal,move,sp,ts,ns,offset,scale,mean,stdev):
    signal = signal[sp:]
    signal_len = len(signal)
    if ns == 0:
        ns = signal_len
    signal = signal[ts:ns]
    if len(signal) == 0:
        return None
    signal = np.flip(signal, axis=0)
    signal = (signal + offset) * scale
    signal = (signal - mean) / stdev

    stride = move[0]
    move = move[1:]
    move_idx = np.where(move == 1)[0][1:] * stride
    move_idx = len(signal) - move_idx
    move_idx = np.flip(move_idx, axis=0)
    signal = np.array_split(signal, move_idx)
    if len(signal) == 0:
        return None
    return signal


def write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict, max_mb):
    save_idx += 1
    df_size = sys.getsizeof(signal_df) / (1024 ** 2)
    if df_size > max_mb and len(signal_df) > 1:
        ## Split the dataframe
        split_num = min(int(df_size // max_mb) + 1 ,len(signal_df))
        split_size = len(signal_df) // split_num
        for split_idx in range(split_num):
            split_df = signal_df.iloc[split_idx * split_size:min((split_idx + 1) * split_size, len(signal_df))].copy()
            save_idx = write_df(split_df, signal_df_path, pid, pod5_idx, save_idx, index_dict, max_mb)
    else:
        save_path = f"{signal_df_path}/{pid}-{pod5_idx}-{save_idx}.pkl"
        if os.path.exists(save_path):
            raise FileExistsError(f"File {save_path} already exists")
        signal_df.to_pickle(save_path)
        id_list = signal_df["read_id"].tolist()
        index_dict[save_path] = id_list
    return save_idx


def extract_signal_proc(pod5_path_list, signal_df_path, pid, index_list, chunk, max_mb, min_mb):
    index_dict_local = {}
    chunk_buffer = []
    pod5_idx = 0

    for pod5_idx, pod5_path in tqdm.tqdm(enumerate(pod5_path_list), total=len(pod5_path_list), desc=f"Parsing POD5 Files"):
        oom_killer()
        signal_list = []
        offset_list = []
        scale_list = []
        id_list = []
        try:
            with pod5.Reader(pod5_path) as reader:
                skipped = 0
                for record in reader:
                    try:
                        signal_arr = record.signal
                        offset = record.calibration.offset
                        scale = record.calibration.scale
                        id = str(record.read_id)
                    except:
                        skipped += 1
                        continue

                    offset_list.append(offset)
                    scale_list.append(scale)
                    signal_list.append(signal_arr)
                    id_list.append(id)

            if skipped > 0:
                printmessage(f"Skipped {skipped} faulty records in: {pod5_path}", msg_type="warning")

        except:
            ## Pod5 file is corrupted
            printmessage(f"Corrupted POD5 file: {pod5_path} - Skipping", msg_type="warning")
            continue

        df = pd.DataFrame({"signal": signal_list, "read_id": id_list, "offset": offset_list, "scale": scale_list})
        del signal_list, offset_list, scale_list, id_list
        gc.collect()

        save_idx = 0

        for chunk_idx in range(0, len(df) // chunk + 1):
            signal_df = df.iloc[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len(df))].copy()
            df_size = sys.getsizeof(signal_df) / (1024 ** 2)
            if len(signal_df) == chunk and df_size > min_mb:
                save_idx = write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
            else:
                chunk_buffer.append(signal_df)
            gc.collect()

        if len(chunk_buffer) > 0:
            signal_df = pd.concat(chunk_buffer, ignore_index=True)
            df_size = sys.getsizeof(signal_df) / (1024 ** 2)
            if df_size > min_mb:
                chunk_buffer = []
                write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
            else:
                chunk_buffer = [signal_df]
        gc.collect()

    if len(chunk_buffer) > 0:
        signal_df = pd.concat(chunk_buffer, ignore_index=True)
        save_idx = 0
        pod5_idx += 1
        write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
        gc.collect()

    index_list.append(index_dict_local)

    return None


def preprocess_pod5(pod5_path, save_path, ncpu, chunk, max_mb, min_mb):
    # Export pod5 to csv
    pod5_path_list = glob.glob(pod5_path + "/*.pod5")
    proc_list = []
    np.random.shuffle(pod5_path_list)
    pod5_path_list_split = np.array_split(pod5_path_list, ncpu)

    man = mp.Manager()
    index_list = man.list()

    for pid in range(ncpu):
        proc = mp.Process(target=extract_signal_proc, args=(pod5_path_list_split[pid], save_path, pid, index_list,
                                                            chunk, max_mb, min_mb))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    index_dict = {}
    for local_index_dict in index_list:
        index_dict.update(local_index_dict)

    man.shutdown()
    gc.collect()

    return index_dict


def segmented_signal_to_block(signal_segmented, segment_len_arr, kmer, sampling, sig_window):
    kmer_pad = (kmer-1)//2
    lr_pad = (sig_window-1)//2
    l_skip = np.sum(segment_len_arr[:kmer_pad])-lr_pad
    r_skip = np.sum(segment_len_arr[-kmer_pad:])-lr_pad
    signal_segmented = np.concatenate(signal_segmented)
    len_seg = len(signal_segmented)

    if len_seg % sampling != 0:
        return None

    signal_segmented = np.reshape(signal_segmented, (-1, sampling))
    signal_segmented = np.concatenate([signal_segmented[i:-sig_window+i] for i in range(sig_window)], axis=1)

    if l_skip > 0:
        signal_segmented = signal_segmented[l_skip:]
    if r_skip > 0:
        signal_segmented = signal_segmented[:-r_skip]

    return signal_segmented


def extract_move(bam_path, ncpu, bq_cutoff, signal_path_dict, signal_path_arr, intermediate_path):
    ## Extract mv tag from bam and save to separate file
    data_dict = {x: {"mv": [], "read_id": [], "ts": [], "ns": [], "sp": [], "seq": [], "bq": [],
                     "ref": [], "start": [], "cigar": []} for x in signal_path_arr}
    valid_count = 0
    missing_move = 0
    missing_bq = 0
    low_bq = 0
    missing_signal = 0
    unmapped = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=ncpu) as input_bam:
        with tqdm.tqdm(total=input_bam.mapped + input_bam.unmapped, desc="Parsing BAM File") as pbar:
            for read in input_bam:
                pbar.update(1)
                pbar.set_postfix({"valid": valid_count, "invalid": missing_move + missing_bq + low_bq + missing_signal + unmapped})
                if read.is_unmapped:
                    unmapped += 1
                    continue

                if read.has_tag("pi"):
                    read_id = str(read.get_tag("pi"))
                else:
                    read_id = str(read.query_name)

                try:
                    bq = np.array(read.query_qualities, dtype=int)
                    if mean_phred(bq) < bq_cutoff:
                        low_bq += 1
                        continue
                except:
                    missing_bq += 1
                    continue

                try:
                    signal_path = signal_path_dict[read_id]
                    data = data_dict[signal_path]
                except:
                    missing_signal += 1
                    continue

                if read.has_tag("mv"):
                    mv = read.get_tag("mv")
                else:
                    missing_move += 1
                    continue

                if read.has_tag("ts"):
                    ts = read.get_tag("ts")
                else:
                    ts = 0

                if read.has_tag("ns"):
                    ns = read.get_tag("ns")
                else:
                    ns = 0

                if read.has_tag("sp"):
                    sp = read.get_tag("sp")
                else:
                    sp = 0

                data["mv"].append(mv)
                data["read_id"].append(read_id)
                data["ts"].append(ts)
                data["ns"].append(ns)
                data["sp"].append(sp)
                data["seq"].append(str(read.query_sequence))
                data["bq"].append(bq)
                data["ref"].append(read.reference_name)
                data["start"].append(read.reference_start)
                data["cigar"].append(read.cigarstring)

                valid_count += 1

    printmessage(f"Valid reads: {valid_count}", msg_type="info")
    printmessage(f"Low BQ: {low_bq}", msg_type="warning")
    printmessage(f"Missing BQ: {missing_bq}", msg_type="warning")
    printmessage(f"Missing Signal: {missing_signal}", msg_type="warning")
    printmessage(f"Missing Move: {missing_move}", msg_type="warning")
    printmessage(f"Unmapped: {unmapped}", msg_type="warning")

    for signal_path, data in tqdm.tqdm(data_dict.items(), total=len(data_dict), desc="Saving Move Data"):
        move_df = pd.DataFrame.from_dict(data, orient="columns")
        df_len = len(move_df)
        if df_len > 0:
            move_df.to_pickle(f"{intermediate_path}/move_df_split/{signal_path}")
        del move_df

    del data_dict

    gc.collect()
    return None


def segment_normalize_signal(seg_df_path, signal_path_arr, norm_factor, label_df, pid,
                             kmer = 5, cb_len = 21, sampling = 6,
                             sig_window = 5, chunk_size = 10000, max_token_len = 200, boi = "A"):

    mean, stdev = norm_factor
    cb_half_len = cb_len//2
    cb_lr_pad = (cb_len-kmer)//2
    trim = kmer//2
    buffer = []

    for signal_path in tqdm.tqdm(signal_path_arr):
        oom_killer()

        out_path = f"{seg_df_path}/token_light_v3/{signal_path.split('/')[-1]}"
        if os.path.exists(out_path):
            printmessage(f"Already exists: {out_path}", msg_type="warning")
            continue
        move_path = f"{seg_df_path}/intermediates/move_df_split/{signal_path.split('/')[-1]}"
        if not os.path.exists(move_path):
            printmessage(f"Move data not found: {move_path}", msg_type="warning")
            continue
        signal_df = pd.read_pickle(signal_path)
        if len(signal_df) == 0:
            printmessage(f"Empty signal data: {signal_path}", msg_type="warning")
            continue

        move_df = pd.read_pickle(move_path)
        signal_df = signal_df.merge(move_df, on="read_id", how="inner")
        del move_df
        gc.collect()

        alignment_zip = zip(signal_df["ref"], signal_df["start"], signal_df["cigar"])
        pos_list = [get_label_pos_list(ref, start, cigar, label_df) for ref, start, cigar in alignment_zip]
        signal_df["pos"] = pos_list
        signal_df = signal_df[signal_df["pos"].apply(lambda x: len(x) > 0)]
        if len(signal_df) == 0:
            print(f"Empty signal data: {signal_path}")
            continue
        del pos_list, alignment_zip
        gc.collect()

        signal_df["mv"] = signal_df["mv"].apply(lambda x: np.array(x, dtype=int))
        signal_df["signal"] = signal_df.apply(lambda x: trim_scale_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"],
                                                                                  x["offset"], x["scale"], mean, stdev), axis=1)
        signal_df = signal_df[["read_id", "ref", "bq", "seq", "signal", "pos"]].copy()
        gc.collect()

        signal_df = signal_df[signal_df["pos"].apply(lambda x: len(x) > 0)]
        signal_df = signal_df.explode("pos").reset_index(drop=True)

        if len(signal_df) == 0:
            print(f"Empty signal data (EXPLODE): {signal_path}")
            continue

        signal_df["ref"] = signal_df["ref"].str.split(".").str[0]
        signal_df["ref_pos"] = signal_df["pos"].apply(lambda x: x[0])
        signal_df["query_pos"] = signal_df["pos"].apply(lambda x: x[1])
        signal_df["label"] = signal_df["pos"].apply(lambda x: x[2])

        signal_df["centre_nuc"] = signal_df.apply(lambda x: x["seq"][x["query_pos"]] if x["query_pos"] < len(x["seq"]) else None, axis=1)
        signal_df = signal_df[signal_df["centre_nuc"] == boi]

        if len(signal_df) == 0:
            print(f"Empty signal data (BOI): {signal_path}")
            continue

        signal_df["label_id"] = signal_df["ref"].astype(str) + ":" + signal_df["ref_pos"].astype(str)
        signal_df["start_pos"] = signal_df["query_pos"] - cb_half_len
        signal_df["end_pos"] = signal_df["query_pos"] + cb_half_len + 1
        signal_df["query_len"] = signal_df["seq"].apply(len)

        signal_df = signal_df[(signal_df["start_pos"] >= 0) & (signal_df["end_pos"] <= signal_df["query_len"])]

        if len(signal_df) == 0:
            print(f"Empty signal data (CB): {signal_path}")
            continue

        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["bq"] = signal_df.apply(lambda x: x["bq"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["motif"] = signal_df.apply(lambda x: x["seq"][x["start_pos"]:x["end_pos"]], axis=1)
        #
        # signal_df["segment_len_arr"] = signal_df["signal"].apply(lambda x: create_segment_len_arr(x, sampling))
        # signal_df["token_len"] = signal_df["segment_len_arr"].apply(lambda x: np.sum(x[trim:-trim]))
        # signal_df = signal_df[(signal_df["segment_len_arr"].apply(lambda x: len(x)==cb_len)) &
        #                       (signal_df["token_len"] <= max_token_len)].copy()
        #
        # try:
        #     signal_df["signal_token"] = signal_df.apply(lambda x: segmented_signal_to_block(x["signal"], x["segment_len_arr"],
        #                                                                                     kmer, sampling, sig_window), axis=1)
        # except:
        #     print(f"Signal Tokenization Error in: {signal_path} - Skipping")
        #     continue
        #
        # signal_df = signal_df[signal_df["signal_token"].notnull()]
        #
        # if len(signal_df) == 0:
        #     continue
        #
        # signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x[trim:-trim])
        # signal_df["bq_token"] = signal_df["bq"].apply(lambda x: x[trim:-trim])
        # signal_df["kmer_token"]  = signal_df["motif"].apply(lambda x: sequence_to_kmer_token(x, kmer))

        # signal_df["kmer_token"] = signal_df.apply(lambda x: expand_token_to_segment(x["kmer_token"], x["segment_len_arr"]), axis=1)
        # signal_df["bq_token"] = signal_df.apply(lambda x: expand_token_to_segment(x["bq"].astype(np.uint8), x["segment_len_arr"]), axis=1)
        # signal_df["move_token"] = signal_df["segment_len_arr"].apply(lambda x: create_move_token(x))
        # signal_df["target_mask"] = signal_df["segment_len_arr"].apply(lambda x: create_target_mask(x, cb_lr_pad))
        # signal_df = signal_df[["label_id", "label", "kmer_token", "bq_token", "signal_token", "move_token", "target_mask"]].copy()

        signal_df = signal_df[["label_id", "label", "signal", "bq", "motif"]].copy()

        gc.collect()

        if len(buffer) > 0:
            signal_df = pd.concat([buffer, signal_df], ignore_index=True)
            buffer = []

        if len(signal_df) < chunk_size:
            buffer = signal_df

        else:
            for chunk_idx in range(0, len(signal_df) // chunk_size):
                chunk = signal_df.iloc[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                chunk.to_pickle(f"{out_path.split('.')[0]}-{chunk_idx}.pkl")

            chunk = signal_df.iloc[(len(signal_df) // chunk_size) * chunk_size:].copy()
            if len(chunk) > 0:
                buffer = chunk
            del signal_df

        gc.collect()


    out_path = f"{seg_df_path}/token_light_v3/last-{pid}.pkl"
    ## For last chunk, append zero data and save
    if len(buffer) > 0:
        for chunk_idx in range(0, len(buffer) // chunk_size):
            chunk = buffer.iloc[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
            chunk.to_pickle(f"{out_path.split('.')[0]}-{chunk_idx}.pkl")

        chunk = buffer.iloc[(len(buffer) // chunk_size) * chunk_size:].copy()
        if len(chunk) > 0:
            chunk.to_pickle(f"{out_path.split('.')[0]}-last.pkl")

    return None


INCREMENTS_CIGAR = {
    'M': [1, 1],
    'I': [0, 1],
    'S': [0, 1],
    'D': [1, 0],
    'N': [1, 0],
}

def ref_pos_to_query_pos(ref_pos_arr, cigar, start_pos):
    x = [[start_pos - 1, -1]]
    for length, op in re.findall(r'(\d+)([MIDSN])', cigar):
        x += [INCREMENTS_CIGAR[op]] * int(length)

    x = np.array(x)
    x = np.cumsum(x, axis=0)[np.sum(x, axis=1) == 2]
    query_pos_list = []

    ref_pos_idx = 0
    ref_pos_idx_max = len(ref_pos_arr)
    for row in x:
        ref_pos_now = ref_pos_arr[ref_pos_idx]
        if row[0] == ref_pos_now:
            query_pos_list.append(row[1])
            if ref_pos_idx < ref_pos_idx_max - 1:
                ref_pos_idx += 1
            else:
                break
        elif row[0] > ref_pos_now:
            query_pos_list.append(None)
            if ref_pos_idx < ref_pos_idx_max - 1:
                ref_pos_idx += 1
            else:
                break

    return query_pos_list


def get_label_pos_list(ref, start, cigar, label_df):
    ref = ref.split(".")[0]

    try:
        label_df_nmid = label_df.get_group(ref)
    except KeyError:
        return []

    label_list = label_df_nmid["label"].values
    ref_pos_list = label_df_nmid["pos"].values
    query_pos_list = ref_pos_to_query_pos(ref_pos_list, cigar, start)
    pos_tuple_list = list(zip(ref_pos_list, query_pos_list, label_list))
    pos_tuple_list_filtered = [x for x in pos_tuple_list if x[1] is not None]
    return pos_tuple_list_filtered


def parse_args():
    ## Usage: "python segment_normalize_signal.py --cpu {args.thread} --pod5 {pod5_path} --bam {bam_path} --block {block_path} --output {signal_path}"
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--pod5", "-p", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--bam", "-b", type=str, required=True, help="Dorado BAM file")
    parser.add_argument("--qcut", "-q", type=int, default=0, help="BQ cutoff")
    parser.add_argument("--wdir", "-w", type=str, default=None, help="Working directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk", "-k", type=int, default=1000, help="Chunk size")
    parser.add_argument("--label", "-l", type=str, required=True, help="Label file")
    parser.add_argument("--max_size", "-m", type=int, default=20, help="Maximum POD5 dataframe size in MB")
    parser.add_argument("--min_size", "-i", type=int, default=10, help="Minimum POD5  dataframe size in MB")
    parser.add_argument("--toml", "-t", type=str, default="/extdata3/baeklab/Hyeonseo/bin/dorado-0.7.0/model/rna004_130bps_sup@v5.0.0/config.toml", help="Dorado Model TOML file")
    args = parser.parse_args()
    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    if not os.path.exists(args.bam):
        raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    # if os.path.exists(args.output) and len(os.listdir(args.output)) > 0:
    #     raise FileExistsError(f"Output directory {args.output} already exists and is not empty")
    if args.wdir is None:
        args.wdir = f"{args.output}/intermediates/"
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.wdir, exist_ok=True)
    return args


def parse_toml(toml_path):
    if not os.path.exists(toml_path):
        printmessage(f"TOML file {toml_path} does not exist", msg_type="error", error=FileNotFoundError)
    toml_dict = toml.load(toml_path)
    if "standardisation" not in toml_dict:
        printmessage("Standardisation section not found in the TOML file. Check Dorado model version.", msg_type="error", error=ValueError)
    std_dict = toml_dict["standardisation"]
    if not std_dict["standardise"]:
        printmessage("Standardisation is not enabled in the TOML file. Check Dorado model version.", msg_type="error", error=ValueError)
    norm_factor = (std_dict["mean"], std_dict["stdev"])
    printmessage(f"Standardisation values loaded from TOML: {norm_factor}", msg_type="info")
    return norm_factor


def main():
    args = parse_args()

    token_output_path = f"{args.output}/token_light_v3/"
    intermediate_path = f"{args.output}/intermediates/"
    signal_raw_path = f"{intermediate_path}/signal_raw/"
    signal_index_path = f"{intermediate_path}/signal_index.pkl"

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(token_output_path, exist_ok=True)
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(signal_raw_path, exist_ok=True)
    os.makedirs(f"{intermediate_path}/move_df_split", exist_ok=True)
    if args.toml is not None:
        norm_factor = parse_toml(args.toml)
    else:
        printmessage("No TOML file provided. Using default values for standardisation.", msg_type="warning")
        norm_factor = (80.8758975922949, 17.26975967138176) ## Default values for rna004_130bps_sup@v5.0.0_m6A@v1

    # index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.cpu, args.chunk, args.min_size, args.max_size)
    # signal_path_arr = list(index_dict.keys())
    # gc.collect()

    # with open(signal_index_path, "wb") as outfile:
    #     pickle.dump(index_dict, outfile)
    # gc.collect()
    #

    # signal_path_dict = {}
    # for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Read-to-File Index"):
    #     for read_id in id_list:
    #         signal_path_dict[read_id] = signal_path.split('/')[-1]
    # del index_dict
    # gc.collect()

    with open(signal_index_path, "rb") as infile:
        index_dict = pickle.load(infile)

    signal_path_dict = {}
    for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Read-to-File Index"):
        for read_id in id_list:
            signal_path_dict[read_id] = signal_path.split('/')[-1]
    signal_path_arr = list(index_dict.keys())

    signal_name_arr = [x.split('/')[-1] for x in signal_path_arr]
    # extract_move(args.bam, args.cpu, args.qcut, signal_path_dict, signal_name_arr, intermediate_path)

    del signal_path_dict, index_dict
    gc.collect()

    label_df = pd.read_csv(args.label, sep='\t')

    np.random.shuffle(signal_path_arr)
    signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.cpu))

    proc_list = []
    label_df = label_df.groupby("nmid")
    for pid, signal_paths in enumerate(signal_path_arr_split):
        proc = mp.Process(target=segment_normalize_signal,

                          args=(args.output, signal_paths, norm_factor, label_df, pid))
        proc_list.append(proc)
        proc.start()

    del signal_path_arr_split
    gc.collect()

    for proc in proc_list:
        proc.join()

    gc.collect()

    return None


if __name__ == "__main__":
    main()
