import argparse, gc, os, sys, pickle, re, pod5, pysam, tqdm, glob
import multiprocessing as mp
import numpy as np
import pandas as pd
from utils.utils import mean_phred, oom_killer, printmessage
import math

def segmented_signal_to_block(signal_segmented, segment_len_arr, kmer, sampling, sig_window, pad_to):
    """
    Segments the signal into blocks and pads it to the specified length.

    Args:
        signal_segmented (np.ndarray): The segmented signal.
        segment_len_arr (np.ndarray): Array of segment lengths.
        kmer (int): K-mer size.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
        pad_to (int): Length to pad the signal to.

    Returns:
        np.ndarray: The segmented and padded signal.
    """
    try:
        kmer_pad = (kmer-1)//2
        lr_pad = (sig_window-1)//2
        l_skip = (np.sum(segment_len_arr[:kmer_pad])-lr_pad)*sampling
        r_skip = (np.sum(segment_len_arr[-kmer_pad:])-lr_pad)*sampling
        assert l_skip >= 0, f"Left skip is negative: {l_skip}, segment_len_arr: {segment_len_arr}"
        assert r_skip >= 0, f"Right skip is negative: {r_skip}, segment_len_arr: {segment_len_arr}"
        signal_segmented = np.concatenate(signal_segmented)
        if len(signal_segmented) % sampling != 0:
            return None
        if r_skip > 0:
            signal_segmented = signal_segmented[l_skip:-r_skip]
        else:
            signal_segmented = signal_segmented[l_skip:]

        padding = (pad_to+kmer-1) * sampling - len(signal_segmented)
        if padding > 0:
            signal_segmented = np.pad(signal_segmented, (0, padding), mode="constant", constant_values=0)

    except:
        return None
    return signal_segmented


def create_segment_len_arr(segment_arr, sampling):
    """
    Creates an array of segment lengths.

    Args:
        segment_arr (list): List of segments.
        sampling (int): Sampling rate.

    Returns:
        np.ndarray: Array of segment lengths.
    """
    segment_len_arr = np.array([len(x) for x in segment_arr], dtype=int)
    segment_len_arr = segment_len_arr // sampling
    return segment_len_arr


def move_to_dwell(move, quantile_a, quantile_b, shift_mult, scale_mult):
    """
    Converts move array to dwell time.

    Args:
        move (np.ndarray): Move array.
        quantile_a (float): Quantile A for normalization.
        quantile_b (float): Quantile B for normalization.
        shift_mult (float): Shift multiplier for normalization.
        scale_mult (float): Scale multiplier for normalization.

    Returns:
        np.ndarray: Dwell time array.
    """
    sampling = move[0]
    move = np.flip(move[1:]) * np.arange(1, len(move))
    move = move[move > 0]
    move = np.concatenate([np.zeros(1, dtype=int), move])
    move = move[1:] - move[:-1]
    move = move * sampling
    move = np.log10(move.astype(np.float32))
    quantile_a_value = np.quantile(move, quantile_a)
    quantile_b_value = np.quantile(move, quantile_b)
    q_shift = max(0.1, shift_mult * (quantile_a_value + quantile_b_value))
    q_scale = max(0.1, scale_mult * (quantile_b_value - quantile_a_value))
    move = (move - q_shift) / q_scale
    return move

def normalise_trim_segment_signal(signal, move, sp, ts, ns, quantile_a, quantile_b, shift_mult, scale_mult):
    """
    Normalizes and trims the signal.

    Args:
        signal (np.ndarray): The signal to normalize and trim.
        move (np.ndarray): Move array.
        sp (int): Start position.
        ts (int): Trim start.
        ns (int): Trim end.
        quantile_a (float): Quantile A for normalization.
        quantile_b (float): Quantile B for normalization.
        shift_mult (float): Shift multiplier for normalization.
        scale_mult (float): Scale multiplier for normalization.

    Returns:
        np.ndarray: The normalized and trimmed signal.
    """
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
    """
    Writes the signal dataframe to a file.

    Args:
        signal_df (pd.DataFrame): Dataframe containing signal data.
        signal_df_path (str): Path to save the dataframe.
        pid (int): Process ID.
        pod5_idx (int): POD5 index.
        save_idx (int): Save index.
        index_dict (dict): Dictionary to store index information.
        max_mb (int): Maximum size of the dataframe in MB.

    Returns:
        int: Updated save index.
    """
    save_idx += 1
    df_size = sys.getsizeof(signal_df) / (1024 ** 2)
    if df_size > max_mb and len(signal_df) > 1:
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

def split_pod5(pod5_dir, max_size_mb, ncpu = 8):
    """
    Splits POD5 files into smaller chunks.

    Args:
        pod5_dir (str): Directory containing POD5 files.
        max_size_mb (int): Maximum size of each chunk in MB.
        ncpu (int): Number of CPUs to use.

    Returns:
        None
    """
    pod5_path_list = glob.glob(pod5_dir + "/*.pod5")
    oversized_list = []
    for pod5_path in pod5_path_list:
        pod5_size = os.path.getsize(pod5_path) / (1024 ** 2)
        if pod5_size > max_size_mb:
            oversized_list.append((pod5_path, pod5_size))
    if len(oversized_list) == 0:
        return None
    else:
        n_proc = min(ncpu, len(oversized_list))
        oversized_list_split = [oversized_list[i::n_proc] for i in range(n_proc)]
        proc_list = []
        for i in range(n_proc):
            proc = mp.Process(target = split_pod5_proc, args = (oversized_list_split[i], max_size_mb))
            proc_list.append(proc)
            proc.start()
        for proc in proc_list:
            proc.join()

    return None

def split_pod5_proc(pod5_list, max_size_mb):
    """
    Splits a list of POD5 files into smaller chunks.

    Args:
        pod5_list (list): List of POD5 files to split.
        max_size_mb (int): Maximum size of each chunk in MB.

    Returns:
        None
    """
    for pod5_path, pod5_size in pod5_list:
        with pod5.Reader(pod5_path) as reader:
            batch_count = reader.batch_count
            writer_count = math.ceil(pod5_size / max_size_mb)
            writer_list = [ pod5.Writer(f"{pod5_path[:-5]}_{x}.pod5") for x in range(writer_count)]
            with tqdm.tqdm(total=batch_count) as pbar:
                for batch_idx, batch in enumerate(reader.read_batches()):
                    writer_list[batch_idx%writer_count].add_reads([x.to_read() for x in batch.reads()])
                    pbar.update(1)
            for writer in writer_list:
                writer.close()
        os.remove(pod5_path)
    return None

def extract_signal_proc(pod5_path_list, signal_df_path, pid, index_list, chunk, max_mb, min_mb):
    """
    Extracts signal data from POD5 files and saves it to a dataframe.

    Args:
        pod5_path_list (list): List of POD5 file paths.
        signal_df_path (str): Path to save the signal dataframe.
        pid (int): Process ID.
        index_list (list): List to store index information.
        chunk (int): Chunk size.
        max_mb (int): Maximum size of the dataframe in MB.
        min_mb (int): Minimum size of the dataframe in MB.

    Returns:
        None
    """
    index_dict_local = {}
    chunk_buffer = None
    pod5_idx = 0

    for pod5_idx, pod5_path in tqdm.tqdm(enumerate(pod5_path_list), total=len(pod5_path_list), desc=f"Parsing POD5 Files"):
        oom_killer()
        signal_list = []
        offset_list = []
        scale_list = []
        id_list = []
        skipped = 0
        save_idx = 0

        try:
            with pod5.Reader(pod5_path) as reader:
                for record_idx, record in enumerate(reader):
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

                    if record_idx % chunk == 0 and record_idx > 0:

                        signal_df = pd.DataFrame({"signal": signal_list, "read_id": id_list, "offset": offset_list, "scale": scale_list})
                        del signal_list, offset_list, scale_list, id_list
                        gc.collect()

                        if chunk_buffer is not None:
                            signal_df = pd.concat([chunk_buffer, signal_df], ignore_index=True)
                            chunk_buffer = None

                        df_size = sys.getsizeof(signal_df) / (1024 ** 2)
                        if df_size > min_mb:
                            save_idx = write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
                            del signal_df

                        else:
                            chunk_buffer = signal_df

                        signal_list = []
                        offset_list = []
                        scale_list = []
                        id_list = []
                        gc.collect()

            signal_df = pd.DataFrame({"signal": signal_list, "read_id": id_list, "offset": offset_list, "scale": scale_list})
            del signal_list, offset_list, scale_list, id_list
            gc.collect()

            if chunk_buffer is not None:
                signal_df = pd.concat([chunk_buffer, signal_df], ignore_index=True)
                chunk_buffer = None

            df_size = sys.getsizeof(signal_df) / (1024 ** 2)
            if df_size > min_mb:
                save_idx = write_df(signal_df, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
                del signal_df
            else:
                chunk_buffer = signal_df
            gc.collect()

            if skipped > 0:
                printmessage(f"Skipped {skipped} faulty records in: {pod5_path}", msg_type="warning")

        except:
            printmessage(f"Corrupted POD5 file: {pod5_path} - Skipping", msg_type="warning")
            continue

    if chunk_buffer is not None:
        save_idx = 0
        pod5_idx += 1
        write_df(chunk_buffer, signal_df_path, pid, pod5_idx, save_idx, index_dict_local, max_mb)
        del chunk_buffer
        gc.collect()

    index_list.append(index_dict_local)

    return None

def preprocess_pod5(pod5_path, save_path, ncpu, chunk, max_mb, min_mb):
    """
    Preprocesses POD5 files by splitting and extracting signal data.

    Args:
        pod5_path (str): Path to the POD5 files.
        save_path (str): Path to save the processed data.
        ncpu (int): Number of CPUs to use.
        chunk (int): Chunk size.
        max_mb (int): Maximum size of the dataframe in MB.
        min_mb (int): Minimum size of the dataframe in MB.

    Returns:
        dict: Dictionary containing index information.
    """
    max_pod5_mb = 4000
    split_pod5(pod5_path, max_pod5_mb, ncpu)

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

def extract_move(bam_path, ncpu, bq_cutoff, signal_path_dict, signal_path_arr, intermediate_path):
    """
    Extracts move data from BAM files and saves it to separate files.

    Args:
        bam_path (str): Path to the BAM file.
        ncpu (int): Number of CPUs to use.
        bq_cutoff (int): Base quality cutoff.
        signal_path_dict (dict): Dictionary mapping read IDs to signal paths.
        signal_path_arr (list): List of signal paths.
        intermediate_path (str): Path to save the intermediate data.

    Returns:
        None
    """
    data_dict = {x: [] for x in signal_path_arr}
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
                except:
                    missing_signal += 1
                    continue

                if not read.has_tag("mv"):
                    missing_move += 1
                    continue

                data_dict[signal_path].append(read)

                valid_count += 1

    printmessage(f"Valid reads: {valid_count}", msg_type="info")
    printmessage(f"Low BQ: {low_bq}", msg_type="info")
    printmessage(f"Missing BQ (Secondary): {missing_bq}", msg_type="info")
    printmessage(f"Missing Signal: {missing_signal}", msg_type="info")
    printmessage(f"Missing Move: {missing_move}", msg_type="info")
    printmessage(f"Unmapped: {unmapped}", msg_type="info")

    for signal_path, data in tqdm.tqdm(data_dict.items(), total=len(data_dict), desc="Saving Move Data"):

        df_dict = {"mv": [], "read_id": [], "ts": [], "ns": [], "sp": [], "seq": [], "bq": [], "pt": [],
                   "ref": [], "start": [], "cigar": []}

        for read in data:

            if read.has_tag("pi"):
                read_id = str(read.get_tag("pi"))
            else:
                read_id = str(read.query_name)

            bq = np.array(read.query_qualities, dtype=int)
            mv = read.get_tag("mv")

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

            df_dict["mv"].append(mv)
            df_dict["read_id"].append(read_id)
            df_dict["ts"].append(ts)
            df_dict["ns"].append(ns)
            df_dict["sp"].append(sp)
            df_dict["pt"].append(pt)
            df_dict["seq"].append(str(read.query_sequence))
            df_dict["bq"].append(bq)
            df_dict["ref"].append(read.reference_name)
            df_dict["start"].append(read.reference_start)
            df_dict["cigar"].append(read.cigarstring)

        move_df = pd.DataFrame.from_dict(df_dict, orient="columns")
        df_len = len(move_df)
        if df_len > 0:
            move_df.to_pickle(f"{intermediate_path}/move_df_split/{signal_path}")
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
    """
    Maps reference positions to query positions based on the CIGAR string.

    Args:
        ref_pos_request_arr (list): List of reference positions to map.
        cigar (str): CIGAR string representing the alignment.
        start_pos (int): Starting position of the alignment.

    Returns:
        list: List of query positions corresponding to the reference positions.
    """
    ## This function strictly assumes that the cigar string is well-formed and ref_pos_request_arr is sorted.
    ## The ref_pos_request_arr will not be sorted in this function, since it corresponds to query_post_list return.

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
    """
    Gets the list of label positions for a given reference, start position, and CIGAR string.

    Args:
        ref (str): Reference sequence name.
        start (int): Start position of the alignment.
        cigar (str): CIGAR string representing the alignment.
        label_df (pd.DataFrame): DataFrame containing label information.

    Returns:
        list: List of tuples containing index and query positions.
    """
    try:
        label_df_nmid = label_df[ref.split(".")[0]]
    except KeyError:
        return []

    index_list = label_df_nmid[0]
    ref_pos_list = label_df_nmid[1]

    ## assert sorted
    if not np.all(np.diff(ref_pos_list) >= 0):
        printmessage("Warning: ref_pos_request_arr is not sorted. It may degrade performance.", msg_type="warning")
        ## sort both lists
        sort_ind = np.argsort(ref_pos_list)
        ref_pos_list = ref_pos_list[sort_ind]
        index_list = index_list[sort_ind]

    query_pos_list = ref_pos_to_query_pos(ref_pos_list, cigar, start)
    pos_tuple_list = list(zip(index_list, query_pos_list))
    pos_tuple_list_filtered = [x for x in pos_tuple_list if x[1] is not None]
    return pos_tuple_list_filtered


def segment_normalize_signal(seg_df_path, signal_path_arr, norm_factor, label_df, pid, token_output_path,
                             cb_len=21, kmer_len=5, chunk_size=10000, max_token_len=200, sampling=6,
                             boi="A", dwell_shift=10, sig_window=5):
    """
    Segments and normalizes the signal data.

    Args:
        seg_df_path (str): Path to the segment data frame.
        signal_path_arr (list): List of signal paths.
        norm_factor (dict): Dictionary containing normalization factors.
        label_df (pd.DataFrame): DataFrame containing label information.
        pid (int): Process ID.
        token_output_path (str): Path to save the tokenized output.
        cb_len (int, optional): Context block length. Defaults to 21.
        kmer_len (int, optional): K-mer length. Defaults to 5.
        chunk_size (int, optional): Chunk size. Defaults to 10000.
        max_token_len (int, optional): Maximum token length. Defaults to 200.
        sampling (int, optional): Sampling rate. Defaults to 6.
        boi (str, optional): Base of interest. Defaults to "A".
        dwell_shift (int, optional): Dwell shift. Defaults to 10.
        sig_window (int, optional): Signal window. Defaults to 5.

    Returns:
        None
    """
    trim = kmer_len // 2

    shift_mult = norm_factor["shift_mult"]
    scale_mult = norm_factor["scale_mult"]
    quantile_a = norm_factor["quantile_a"]
    quantile_b = norm_factor["quantile_b"]

    cb_half_len = cb_len // 2
    buffer = []

    for signal_path in tqdm.tqdm(signal_path_arr):
        oom_killer()

        out_path = f"{token_output_path}/{signal_path.split('/')[-1]}"
        if os.path.exists(out_path):
            printmessage(f"Already exists: {out_path}", msg_type="warning")
            continue
        move_path = f"{seg_df_path}/intermediates/move_df_split/{signal_path.split('/')[-1]}"
        if not os.path.exists(move_path):
            continue
        signal_df = pd.read_pickle(signal_path)
        if len(signal_df) == 0:
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
            continue
        del pos_list, alignment_zip
        gc.collect()

        signal_df["mv"] = signal_df["mv"].apply(lambda x: np.array(x, dtype=int))
        signal_df["dwell_token"] = signal_df["mv"].apply(lambda x: move_to_dwell(x, 0.2, 0.8, 0.5, 1.5))

        signal_df["signal"] = signal_df.apply(lambda x: normalise_trim_segment_signal(x["signal"], x["mv"], x["sp"], x["ts"], x["ns"],
                                                                                      quantile_a, quantile_b, shift_mult, scale_mult), axis=1)

        signal_df = signal_df[["bq", "seq", "signal", "dwell_token", "pos"]].copy()
        gc.collect()

        signal_df = signal_df[signal_df["pos"].apply(lambda x: len(x) > 0)]
        signal_df = signal_df.explode("pos").reset_index(drop=True)

        if len(signal_df) == 0:
            continue

        signal_df["label_id"] = signal_df["pos"].apply(lambda x: x[0])
        signal_df["query_pos"] = signal_df["pos"].apply(lambda x: x[1])
        signal_df["centre_nuc"] = signal_df.apply(lambda x: x["seq"][x["query_pos"]] if x["query_pos"] < len(x["seq"]) else None, axis=1)

        signal_df = signal_df[signal_df["centre_nuc"] == boi]

        if len(signal_df) == 0:
            continue

        signal_df["start_pos"] = signal_df["query_pos"] - cb_half_len
        signal_df["end_pos"] = signal_df["query_pos"] + cb_half_len + 1
        signal_df["query_len"] = signal_df["seq"].apply(len)

        signal_df = signal_df[(signal_df["start_pos"] >= 0) & (signal_df["end_pos"] + dwell_shift - trim < signal_df["query_len"])]
        signal_df.dropna(inplace=True)

        if len(signal_df) == 0:
            continue

        signal_df["signal"] = signal_df.apply(lambda x: x["signal"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["dwell_motor_token"] = signal_df.apply(lambda x: x["dwell_token"][(x["start_pos"] + dwell_shift + trim):(x["end_pos"] + dwell_shift - trim)], axis=1)
        signal_df["dwell_pore_token"] = signal_df.apply(lambda x: x["dwell_token"][(x["start_pos"] + trim):(x["end_pos"] - trim)], axis=1)
        signal_df["bq"] = signal_df.apply(lambda x: x["bq"][x["start_pos"] + trim:x["end_pos"] - trim], axis=1)
        signal_df["motif"] = signal_df.apply(lambda x: x["seq"][x["start_pos"]:x["end_pos"]], axis=1)
        signal_df["segment_len_arr"] = signal_df["signal"].apply(lambda x: create_segment_len_arr(x, sampling))

        signal_df["token_len"] = signal_df["segment_len_arr"].apply(lambda x: np.sum(x[trim:-trim]))
        signal_df = signal_df[(signal_df["segment_len_arr"].apply(lambda x: len(x) == cb_len)) &
                              (signal_df["token_len"] <= max_token_len) &
                              (signal_df["token_len"] > 0)]
        try:
            signal_df["signal"] = signal_df.apply(lambda x: segmented_signal_to_block(x["signal"], x["segment_len_arr"],
                                                                                      kmer_len, sampling, sig_window, max_token_len), axis=1)
        except:
            print(f"Signal Tokenization Error in: {signal_path} - Skipping")
            continue

        signal_df = signal_df[signal_df["signal"].notnull()]

        if len(signal_df) == 0:
            continue

        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x[trim:-trim])

        signal_df = signal_df[["segment_len_arr", "signal", "motif", "dwell_motor_token", "dwell_pore_token", "bq", "label_id"]].copy()
        signal_df.rename(columns={"motif": "kmer_token", "signal": "signal_token", "bq": "bq_token"}, inplace=True)
        signal_df["segment_len_arr"] = signal_df["segment_len_arr"].apply(lambda x: x.astype(np.int16))
        signal_df["signal_token"] = signal_df["signal_token"].apply(lambda x: x.astype(np.float32))
        signal_df["kmer_token"] = signal_df["kmer_token"].apply(lambda x: np.array(list(x)).view(np.int32).astype(np.uint8))
        signal_df["bq_token"] = signal_df["bq_token"].apply(lambda x: np.clip(x, 0, 60).astype(np.uint8))

        if len(buffer) > 0:
            signal_df = pd.concat([buffer, signal_df], ignore_index=True)
            buffer = []

        if len(signal_df) < chunk_size:
            buffer = signal_df

        else:
            for chunk_idx in range(0, len(signal_df) // chunk_size):
                chunk = signal_df.iloc[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                outpath = f"{out_path.split('.')[0]}-{chunk_idx}.npz"
                save_npz(outpath, chunk)

            chunk = signal_df.iloc[(len(signal_df) // chunk_size) * chunk_size:].copy()
            if len(chunk) > 0:
                buffer = chunk
            del signal_df

        gc.collect()

    out_path = f"{token_output_path}/last-{pid}.pkl"

    if len(buffer) > 0:
        for chunk_idx in range(0, len(buffer) // chunk_size):
            chunk = buffer.iloc[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
            outpath = f"{out_path.split('.')[0]}-{chunk_idx}.npz"
            save_npz(outpath, chunk)

        chunk = buffer.iloc[(len(buffer) // chunk_size) * chunk_size:].copy()
        if len(chunk) > 0:
            outpath = f"{out_path.split('.')[0]}-last.npz"
            save_npz(outpath, chunk)

    return None
def save_npz(save_path, df):
    """
    Saves the given DataFrame to a compressed NPZ file.

    Args:
        save_path (str): The path where the NPZ file will be saved.
        df (pd.DataFrame): The DataFrame containing the data to be saved.

    Returns:
        None
    """
    segment_len_arr = np.stack(df["segment_len_arr"].values)
    signal_token = np.stack(df["signal_token"].values)
    kmer_token = np.stack(df["kmer_token"].values)
    dwell_motor_token = np.stack(df["dwell_motor_token"].values)
    dwell_pore_token = np.stack(df["dwell_pore_token"].values)
    bq_token = np.stack(df["bq_token"].values)
    label_id = df["label_id"].values
    np.savez_compressed(save_path,
                        segment_len_arr=segment_len_arr,
                        signal_token=signal_token,
                        kmer_token=kmer_token,
                        dwell_motor_token=dwell_motor_token,
                        dwell_pore_token=dwell_pore_token,
                        bq_token=bq_token,
                        label_id=label_id)
    return None


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Segment and Normalize Signal")
    num_cpu = os.cpu_count()
    parser.add_argument("--cpu", "-c", type=int, default=int(num_cpu * 0.9), help="Number of threads")
    parser.add_argument("--pod5", "-p", type=str, default=None, help="POD5 Input directory")
    parser.add_argument("--bam", "-b", type=str, default=None, help="Dorado BAM file")
    parser.add_argument("--qcut", "-q", type=int, default=0, help="BQ cutoff")
    parser.add_argument("--wdir", "-w", type=str, default=None, help="Working directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--chunk", "-k", type=int, default=16000, help="Chunk size")
    parser.add_argument("--pod5_chunk", "-j", type=int, default=1000, help="POD5 Chunk size")
    parser.add_argument("--label", "-l", type=str, required=True, help="Label file")
    parser.add_argument("--max_size", "-m", type=int, default=20, help="Maximum POD5 dataframe size in MB")
    parser.add_argument("--min_size", "-i", type=int, default=10, help="Minimum POD5 dataframe size in MB")
    parser.add_argument("--postfix", "-x", type=str, default="", help="Postfix for output files")
    parser.add_argument("--max_token_len", "-z", type=int, default=200, help="Maximum token length")
    parser.add_argument("--sampling", "-s", type=int, default=6, help="Sampling rate")
    parser.add_argument("--boi", "-y", type=str, default="A", help="Base of interest")
    parser.add_argument("--kmer_len", "-e", type=int, default=5, help="Kmer length")
    parser.add_argument("--cb_len", "-a", type=int, default=21, help="Context block length")
    parser.add_argument("--max_depth", "-d", type=int, default=None, help="Max Depth")
    args = parser.parse_args()
    if not os.path.exists(args.label):
        raise FileNotFoundError(f"Label (SAMtools Pileup) file {args.label} does not exist")
    if args.wdir is None:
        args.wdir = f"{args.output}/intermediates/"
    if not os.path.exists(args.wdir):
        if not os.path.exists(args.pod5):
            raise FileNotFoundError(f"Input directory {args.pod5} does not exist")
        if not os.path.exists(args.bam):
            raise FileNotFoundError(f"BAM file {args.bam} does not exist")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.wdir, exist_ok=True)
    return args


def get_norm_factor():
    """
    Returns the default normalization factors.

    Returns:
        dict: A dictionary containing the default normalization factors.
    """
    norm_factor_default = {}
    norm_factor_default["quantile_a"] = 0.2
    norm_factor_default["quantile_b"] = 0.8
    norm_factor_default["shift_mult"] = 0.48
    norm_factor_default["scale_mult"] = 0.59

    return norm_factor_default

def main():
    """
    Main function to segment and normalize signal data.

    Steps:
        1. Parse command-line arguments.
        2. Set up directories and normalization factors.
        3. Preprocess POD5 files.
        4. Extract move data.
        5. Segment and normalize signal data.

    Returns:
        None
    """
    args = parse_args()

    token_output_path = f"{args.output}/token_{args.postfix}/"
    intermediate_path = f"{args.output}/intermediates/"
    signal_raw_path = f"{intermediate_path}/signal_raw/"
    signal_index_path = f"{intermediate_path}/signal_index.pkl"

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(token_output_path, exist_ok=True)
    os.makedirs(intermediate_path, exist_ok=True)
    os.makedirs(signal_raw_path, exist_ok=True)
    os.makedirs(f"{intermediate_path}/move_df_split", exist_ok=True)

    norm_factor = get_norm_factor()
    printmessage(f"Normalisation factor: {norm_factor}", msg_type="info")

    if not os.path.exists(signal_index_path):
        index_dict = preprocess_pod5(args.pod5, signal_raw_path, args.cpu, args.pod5_chunk, args.min_size, args.max_size)
        gc.collect()

        with open(signal_index_path, "wb") as outfile:
            pickle.dump(index_dict, outfile)
        gc.collect()

        signal_path_dict = {}
        for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Read-to-File Index"):
            for read_id in id_list:
                signal_path_dict[read_id] = signal_path.split('/')[-1]
        signal_path_arr = list(index_dict.keys())

    else:
        printmessage("Signal data already exists. Skipping extraction.", msg_type="info")
        with open(signal_index_path, "rb") as infile:
            index_dict = pickle.load(infile)

        signal_path_dict = {}
        for signal_path, id_list in tqdm.tqdm(index_dict.items(), total=len(index_dict), desc="Creating Read-to-File Index"):
            for read_id in id_list:
                signal_path_dict[read_id] = signal_path.split('/')[-1]
        signal_path_arr = list(index_dict.keys())

    del index_dict
    gc.collect()

    if os.path.exists(f"{intermediate_path}/move_df_split") and len(glob.glob(f"{intermediate_path}/move_df_split/*.pkl")) > 0:
        printmessage("Move data already exists. Skipping extraction.", msg_type="info")

    else:
        signal_name_arr = [x.split('/')[-1] for x in signal_path_arr]
        extract_move(args.bam, args.cpu, args.qcut, signal_path_dict, signal_name_arr, intermediate_path)

    del signal_path_dict
    gc.collect()

    label_df = pd.read_pickle(args.label)
    label_df["index"] = label_df.index.astype(np.int32)
    if args.max_depth is not None:
        label_df = label_df[label_df["depth"] <= args.max_depth]
    label_df = label_df[["ref", "pos", "index"]].copy()
    label_df = label_df.sort_values("pos")
    label_df.rename({"ref":"nmid"}, axis=1, inplace=True)
    label_df["nmid"] = label_df["nmid"].str.split(".").str[0]
    label_df["pos"] = label_df["pos"] - 1
    label_df = label_df.groupby("nmid")
    label_df = {nmid:df[["index","pos"]].values.T for nmid, df in label_df}
    gc.collect()

    signal_path_arr_split = np.array_split(signal_path_arr, max(1, args.cpu))

    proc_list = []
    for pid, signal_paths in enumerate(signal_path_arr_split):
        proc = mp.Process(target=segment_normalize_signal,
                          args=(args.output, signal_paths, norm_factor, label_df, pid, token_output_path,
                                args.cb_len, args.kmer_len, args.chunk, args.max_token_len, args.sampling, args.boi))
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