import argparse, gc, os, sys, pickle, re, pod5, pysam, tqdm, glob, toml
import multiprocessing as mp
import numpy as np
import pandas as pd
from utils.utils import mean_phred, oom_killer, printmessage


import argparse, gc, os, sys, pickle, re, pod5, pysam, tqdm, glob, toml
import multiprocessing as mp
import numpy as np
import pandas as pd
from utils.utils import mean_phred, oom_killer, printmessage



def create_segment_len_arr(segment_arr, sampling):
    segment_len_arr = np.array([len(x) for x in segment_arr], dtype=int)
    segment_len_arr = segment_len_arr // sampling
    return segment_len_arr


def standardise_trim_segment_signal(signal,move,sp,ts,ns,offset,scale,mean,stdev):
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


def normalise_trim_segment_signal(signal,move,sp,ts,ns, quantile_a, quantile_b, shift_mult, scale_mult):
    signal = signal[sp:]
    signal_len = len(signal)
    if ns == 0:
        ns = signal_len
    signal = signal[ts:ns]
    if len(signal) == 0:
        return None
    signal = np.flip(signal, axis=0)

    quantile_a_value = np.quantile(signal, quantile_a)
    quantile_b_value = np.quantile(signal, quantile_b)

    q_shift = max(10.0, shift_mult * (quantile_a_value + quantile_b_value))
    q_scale = max(1.0, scale_mult * (quantile_b_value - quantile_a_value))
    signal = (signal - q_shift) / q_scale

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



def extract_move_master(bam_path, ncpu, bq_cutoff, signal_path_dict, signal_path_arr, intermediate_path):
    proc_list = []
    manager = mp.Manager()
    count_dict = manager.dict()
    for key in ["valid", "low_bq", "missing_bq", "missing_signal", "missing_move", "unmapped"]:
        count_dict[key] = manager.list()

    ncpu = 32

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=32) as input_bam:

        for pid in range(ncpu):
            proc = mp.Process(target=extract_move_worker, args=(input_bam, pid, ncpu, bq_cutoff, signal_path_dict,
                                                                signal_path_arr, intermediate_path, count_dict))
            proc.start()
            proc_list.append(proc)

        for proc in proc_list:
            proc.join()

    gc.collect()

    count_dict = {key: sum(value) for key, value in count_dict.items()}
    manager.shutdown()

    printmessage(f"Valid reads: {count_dict['valid']}", msg_type="info")
    printmessage(f"Low BQ: {count_dict['low_bq']}", msg_type="info")
    printmessage(f"Missing BQ (Secondary): {count_dict['missing_bq']}", msg_type="info")
    printmessage(f"Missing Signal: {count_dict['missing_signal']}", msg_type="info")
    printmessage(f"Missing Move: {count_dict['missing_move']}", msg_type="info")
    printmessage(f"Unmapped: {count_dict['unmapped']}", msg_type="info")

    for signal_path in signal_path_arr:
        move_df_list = glob.glob(f"{intermediate_path}/move_df_split/{signal_path}-*.pkl")
        move_df_list = [pd.read_pickle(x) for x in move_df_list]
        move_df = pd.concat(move_df_list, ignore_index=True)
        move_df.to_pickle(f"{intermediate_path}/move_df/{signal_path}.pkl")
        for file in move_df_list:
            os.remove(file)

    del move_df_list, move_df

    gc.collect()

    return None


def extract_move_worker(input_bam, pid, ncpu, bq_cutoff, signal_path_dict, signal_path_arr, intermediate_path, count_dict):
    ## Extract mv tag from bam and save to separate file
    data_dict = {x: {"mv": [], "read_id": [], "ts": [], "ns": [], "sp": [], "seq": [], "bq": [], "pt": [],
                     "ref": [], "start": [], "cigar": []} for x in signal_path_arr}
    valid_count = 0
    missing_move = 0
    missing_bq = 0
    low_bq = 0
    missing_signal = 0
    unmapped = 0
    total = np.ceil((input_bam.mapped + input_bam.unmapped) / ncpu).astype(int)

    with tqdm.tqdm(total=total, desc="Parsing BAM File") as pbar:
        for read_idx, read in enumerate(input_bam):

            if read_idx % ncpu != pid:
                continue

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

            if read.has_tag("pt"):
                pt = read.get_tag("pt")
            else:
                pt = 0

            data["mv"].append(mv)
            data["read_id"].append(read_id)
            data["ts"].append(ts)
            data["ns"].append(ns)
            data["sp"].append(sp)
            data["pt"].append(pt)
            data["seq"].append(str(read.query_sequence))
            data["bq"].append(bq)
            data["ref"].append(read.reference_name)
            data["start"].append(read.reference_start)
            data["cigar"].append(read.cigarstring)

            valid_count += 1

    count_dict["valid"].append(valid_count)
    count_dict["low_bq"].append(low_bq)
    count_dict["missing_bq"].append(missing_bq)
    count_dict["missing_signal"].append(missing_signal)
    count_dict["missing_move"].append(missing_move)
    count_dict["unmapped"].append(unmapped)

    for signal_path, data in tqdm.tqdm(data_dict.items(), total=len(data_dict), desc="Saving Move Data"):
        move_df = pd.DataFrame.from_dict(data, orient="columns")
        df_len = len(move_df)
        if df_len > 0:
            move_df.to_pickle(f"{intermediate_path}/move_df_split/{signal_path}-{pid}.pkl")
        del move_df

    del data_dict

    gc.collect()
    return None



INCREMENTS_CIGAR = {
    'M': [1, 1],
    'I': [0, 1],
    'S': [0, 1],
    'D': [1, 0],
    'N': [1, 0],
}


def ref_pos_to_query_pos(ref_pos_request_arr, cigar, start_pos):
    ## This function strictly assumes that the cigar string is well-formed and ref_pos_request_arr is sorted.
    ## So better assert it before calling this function.

    x = [[start_pos - 1, -1]]
    for length, op in re.findall(r'(\d+)([MIDSN])', cigar):
        x += [INCREMENTS_CIGAR[op]] * int(length)

    ## Make reference-to-query mapping
    x = np.array(x)
    ## Only keep the matched positions (including mismatches, because this function doesn't receive the md tag)
    ## Mismatches should be handled outside of this function
    x = np.cumsum(x, axis=0)[np.sum(x, axis=1) == 2]
    query_pos_list = []

    ref_pos_request_idx = 0
    ref_pos_request_idx_max = len(ref_pos_request_arr)
    exit_flag = False
    for row in x:
        while True:
            ## I'm pretty confident that it will cause infinite loop one day, but who cares?
            ref_pos_request = ref_pos_request_arr[ref_pos_request_idx]
            ref_pos, query_pos = row
            if ref_pos == ref_pos_request:
                ## Found the requested ref_pos. Register the query_pos. Go fetch next request AND next ref_pos.
                query_pos_list.append(query_pos)
                if ref_pos_request_idx < ref_pos_request_idx_max - 1:
                    ref_pos_request_idx += 1
                    break
                else:
                    exit_flag = True
                    break
            elif ref_pos > ref_pos_request:
                ## Skipped the requested ref_pos. Register NONE. Go fetch next request.
                query_pos_list.append(None)
                if ref_pos_request_idx < ref_pos_request_idx_max - 1:
                    ref_pos_request_idx += 1
                else:
                    exit_flag = True
                    break
            else:
                ## ref_pos < ref_pos_request
                ## Not yet reached the requested ref_pos. Go fetch next ref_pos.
                break

        if exit_flag:
            break

    return query_pos_list


def get_label_pos_list(ref, start, cigar, label_df):
    ref = ref.split(".")[0]

    try:
        label_df_nmid = label_df.get_group(ref)
    except KeyError:
        return []

    ref_pos_list = label_df_nmid["pos"].values
    query_pos_list = ref_pos_to_query_pos(ref_pos_list, cigar, start)
    pos_tuple_list = list(zip(ref_pos_list, query_pos_list))
    pos_tuple_list_filtered = [x for x in pos_tuple_list if x[1] is not None]
    return pos_tuple_list_filtered


def segment_normalize_signal(seg_df_path, signal_path_arr, norm_factor, label_df, pid, norm_mode,
                             cb_len = 21, kmer_len = 5, chunk_size = 10000, max_token_len = 200, sampling = 6, boi = "A"):

    trim = kmer_len//2
    mean, stdev, quantile_a, quantile_b, shift_mult, scale_mult = None, None, None, None, None, None

    if norm_mode == "normalise":
        shift_mult = norm_factor["shift_mult"]
        scale_mult = norm_factor["scale_mult"]
        quantile_a = norm_factor["quantile_a"]
        quantile_b = norm_factor["quantile_b"]
    elif norm_mode == "standardise":
        mean = norm_factor["mean"]
        stdev = norm_factor["stdev"]
    else:
        raise ValueError(f"Invalid norm_mode: {norm_mode}")


    cb_half_len = cb_len//2
    buffer = []

    for signal_path in tqdm.tqdm(signal_path_arr):
        oom_killer()

        out_path = f"{seg_df_path}/token_{norm_mode}/{signal_path.split('/')[-1]}"
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

        if norm_mode == "normalise":
            signal_df["signal"] = signal_df.apply(lambda x: normalise_trim_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"],
                                                                                          quantile_a, quantile_b, shift_mult, scale_mult), axis=1)

        elif norm_mode == "standardise":
            signal_df["signal"] = signal_df.apply(lambda x: standardise_trim_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"],
                                                                                            x["offset"], x["scale"], mean, stdev), axis=1)

        else:
            raise ValueError(f"Invalid norm_mode: {norm_mode}")

        signal_df = signal_df[["read_id", "ref", "bq", "seq", "signal", "pos", "pt"]].copy()
        gc.collect()

        signal_df = signal_df[signal_df["pos"].apply(lambda x: len(x) > 0)]
        signal_df = signal_df.explode("pos").reset_index(drop=True)

        if len(signal_df) == 0:
            print(f"Empty signal data: {signal_path}")
            continue

        signal_df["ref"] = signal_df["ref"].str.split(".").str[0]
        signal_df["ref_pos"] = signal_df["pos"].apply(lambda x: x[0])
        signal_df["query_pos"] = signal_df["pos"].apply(lambda x: x[1])

        signal_df["centre_nuc"] = signal_df.apply(lambda x: x["seq"][x["query_pos"]] if x["query_pos"] < len(x["seq"]) else None, axis=1)
        signal_df = signal_df[signal_df["centre_nuc"] == boi]

        if len(signal_df) == 0:
            print(f"Empty signal data: {signal_path}")
            continue

        signal_df["label_id"] = signal_df["ref"].astype(str) + ":" + signal_df["ref_pos"].astype(str)
        signal_df["block_id"] = signal_df["read_id"] + ":" + signal_df["query_pos"].astype(str)
        signal_df["start_pos"] = signal_df["query_pos"] - cb_half_len
        signal_df["end_pos"] = signal_df["query_pos"] + cb_half_len + 1
        signal_df["query_len"] = signal_df["seq"].apply(len) - signal_df["pt"]

        signal_df = signal_df[(signal_df["start_pos"] >= 0) & (signal_df["end_pos"] <= signal_df["query_len"])]

        if len(signal_df) == 0:
            print(f"Empty signal data: {signal_path}")
            continue

        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["bq"] = signal_df.apply(lambda x: x["bq"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["motif"] = signal_df.apply(lambda x: x["seq"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["segment_len_arr"] = signal_df["signal"].apply(lambda x: create_segment_len_arr(x, sampling))
        signal_df["token_len"] = signal_df["segment_len_arr"].apply(lambda x: np.sum(x[trim:-trim]))
        signal_df = signal_df[(signal_df["segment_len_arr"].apply(lambda x: len(x)==cb_len)) & (signal_df["token_len"] <= max_token_len)]
        signal_df = signal_df[["label_id", "block_id", "signal", "bq", "motif"]].copy()

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


    out_path = f"{seg_df_path}/token_{norm_mode}/last-{pid}.pkl"
    ## For last chunk, append zero data and save
    if len(buffer) > 0:
        for chunk_idx in range(0, len(buffer) // chunk_size):
            chunk = buffer.iloc[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
            chunk.to_pickle(f"{out_path.split('.')[0]}-{chunk_idx}.pkl")

        chunk = buffer.iloc[(len(buffer) // chunk_size) * chunk_size:].copy()
        if len(chunk) > 0:
            chunk.to_pickle(f"{out_path.split('.')[0]}-last.pkl")

    return None


def parse_args():
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
    parser.add_argument("--toml", "-t", type=str, default=None, help="Dorado Model TOML file")
    parser.add_argument("--norm_mode", "-n", type=str, required=True, help="Normalisation mode: normalise or standardise")
    args = parser.parse_args()
    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
    if not os.path.exists(args.bam):
        raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    if args.wdir is None:
        args.wdir = f"{args.output}/intermediates/"
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.wdir, exist_ok=True)
    if args.toml is None:
        if args.norm_mode == "standardise":
            args.toml = "/extdata3/baeklab/Hyeonseo/bin/dorado-0.7.2/model/rna004_130bps_sup@v5.0.0/config.toml"
        elif args.norm_mode == "normalise":
            args.toml = "/extdata3/baeklab/Hyeonseo/bin/dorado-0.4.3/model/rna004_130bps_sup@v3.0.1/config.toml"
        else:
            raise ValueError(f"Invalid norm_mode: {args.norm_mode}")

    return args


def parse_toml(toml_path, norm_mode):

    if norm_mode == "normalise":
        norm_factor_default = {}
        norm_factor_default["quantile_a"] = 0.2
        norm_factor_default["quantile_b"] = 0.8
        norm_factor_default["shift_mult"] = 0.48
        norm_factor_default["scale_mult"] = 0.59

        if not os.path.exists(toml_path):
            printmessage(f"TOML file {toml_path} does not exist", msg_type="warning")
            printmessage("Using default values for standardisation.", msg_type="warning")
            return norm_factor_default

        toml_dict = toml.load(toml_path)
        if "normalisation" not in toml_dict:
            printmessage("normalisation section not found in the TOML file. Check Dorado model version.", msg_type="warning")
            printmessage("Using default values for normalisation.", msg_type="warning")
            return norm_factor_default

        printmessage("Normalisation parameters found in TOML file.", msg_type="info")

        std_dict = toml_dict["normalisation"]
        norm_factor = {}
        norm_factor["quantile_a"] = std_dict.get("quantile_a")
        norm_factor["quantile_b"] = std_dict.get("quantile_b")
        norm_factor["shift_mult"] = std_dict.get("shift_multiplier")
        norm_factor["scale_mult"] = std_dict.get("scale_multiplier")

        ## sanitize
        for key in norm_factor.keys():
            if norm_factor[key] is None:
                printmessage(f"Key {key} not found in TOML file. Falling back to default value.", msg_type="warning")
                norm_factor[key] = norm_factor_default[key]

    elif norm_mode == "standardise":

        norm_factor_default = {}
        norm_factor_default["mean"] = 80.8758975922949
        norm_factor_default["stdev"] = 17.26975967138176

        if not os.path.exists(toml_path):
            printmessage(f"TOML file {toml_path} does not exist", msg_type="warning")
            printmessage("Using default values for standardisation.", msg_type="warning")
            return norm_factor_default

        toml_dict = toml.load(toml_path)
        if "standardisation" not in toml_dict:
            printmessage("standardisation section not found in the TOML file. Check Dorado model version.", msg_type="warning")
            printmessage("Using default values for standardisation.", msg_type="warning")
            return norm_factor_default

        printmessage("Normalisation parameters found in TOML file.", msg_type="info")

        std_dict = toml_dict["standardisation"]

        if not std_dict["standardise"]:
            printmessage("Standardisation is not enabled in the TOML file. Check Dorado model version.", msg_type="warning")
            printmessage("Using default values for standardisation.", msg_type="warning")
            return norm_factor_default

        norm_factor = {}
        norm_factor["mean"] = std_dict.get("mean")
        norm_factor["stdev"] = std_dict.get("stdev")

        ## sanitize
        for key in norm_factor.keys():
            if norm_factor[key] is None:
                printmessage(f"Key {key} not found in TOML file. Falling back to default value.", msg_type="warning")
                norm_factor[key] = norm_factor_default[key]

    else:
        printmessage(f"Mode {norm_mode} not recognised. Should be either 'normalise' or 'standardise'.", msg_type="error")
        raise ValueError()

    return norm_factor

def main():
    args = parse_args()

    token_output_path = f"{args.output}/token_{args.norm_mode}/"
    intermediate_path = f"{args.output}/intermediates/"
    signal_raw_path = f"{intermediate_path}/signal_raw/"
    signal_index_path = f"{intermediate_path}/signal_index.pkl"

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(token_output_path, exist_ok=True)
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(signal_raw_path, exist_ok=True)
    os.makedirs(f"{intermediate_path}/move_df_split", exist_ok=True)

    norm_factor = parse_toml(args.toml, args.norm_mode)
    printmessage(f"Normalisation factor: {norm_factor}", msg_type="info")

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

    # signal_name_arr = [x.split('/')[-1] for x in signal_path_arr]
    # extract_move_worker(args.bam, args.cpu, args.qcut, signal_path_dict, signal_name_arr, intermediate_path)

    del signal_path_dict, index_dict
    gc.collect()

    label_df = pd.read_csv(args.label, sep='\t')
    signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.cpu))

    proc_list = []
    label_df = label_df.groupby("nmid")
    for pid, signal_paths in enumerate(signal_path_arr_split):
        proc = mp.Process(target=segment_normalize_signal,
                          args=(args.output, signal_paths, norm_factor, label_df, pid, args.norm_mode))
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
