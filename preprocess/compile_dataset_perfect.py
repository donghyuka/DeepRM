##
## Output structure:
##  |
##  |---/score-all
##  |   |---/train
##  |   |   |---/pos
##  |   |   |---/neg
##  |   |---/val
##  |   |   |---/pos
##  |   |   |---/neg
##  |
##  |---/score-perfect
##      |---/train
##      |   |---/pos
##      |   |---/neg
##      |---/val
##      |   |---/pos
##      |   |---/neg
##


import numpy as np
import multiprocessing as mp
import os, argparse, tqdm, gc, glob
from utils.utils import printmessage, oom_killer


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--pos", dest="pos_path", type=str, default=None, nargs="+", help="Positive token files")
    args.add_argument("--neg", dest="neg_path", type=str, default=None, nargs="+", help="Negative token files")
    args.add_argument("--out", dest="out_path", type=str, required=True, help="Output directory")
    args.add_argument("--cpu", dest="cpu", type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args.add_argument("--chk", dest="chunk", type=int, default=4000, help="Chunk size")
    args.add_argument("--seed", dest="seed", type=int, default=None, help="Random seed")
    args = args.parse_args()

    if os.path.exists(args.out_path):
        printmessage(f"Output directory {args.out_path} already exists", msg_type="error", error=FileExistsError)
    else:
        os.makedirs(args.out_path)

    if args.pos_path is None:
        printmessage("No positive token files specified", msg_type="warning")
    else:
        for pos_path in args.pos_path:
            if not os.path.exists(pos_path):
                printmessage(f"Positive token file {pos_path} does not exist", msg_type="error", error=FileNotFoundError)

    if args.neg_path is None:
        printmessage("No negative token files specified", msg_type="warning")
    else:
        for neg_path in args.neg_path:
            if not os.path.exists(neg_path):
                printmessage(f"Negative token file {neg_path} does not exist", msg_type="error", error=FileNotFoundError)

    return args



def sample_and_save(in_path_list, out_path, ncpu, label, chunk,
                       label_dict = {0:"neg", 1:"pos"},
                       set_split_dict = {"train":0.95, "val":0.05},
                       score_name_list = ["all", "perfect"],
                       id_digit=9,
                       shuffle = True,
                       read_once = 100):

    in_file_list = [x for in_path in in_path_list for x in glob.glob(f"{in_path}/*.npz")]
    column_keys = ["segment_len_arr", "signal_token", "kmer_token", "dwell_motor_token", "dwell_pore_token", "bq_token"]


    if shuffle:
        in_file_list = np.random.permutation(in_file_list)
    in_file_list = np.array_split(in_file_list, ncpu)
    proc_list = []
    man = mp.Manager()
    remainder_dict = man.dict()
    label_str = label_dict[label]

    for set_name in set_split_dict:
        for score in score_name_list:
            remainder_dict[(set_name,score)] = man.list()

    for pid in range(ncpu):
        proc = mp.Process(target=sample_and_save_worker, args=(ncpu, pid, in_file_list[pid], out_path, label_str,
                                                                  set_split_dict, chunk, label,
                                                                  remainder_dict, id_digit, shuffle, read_once, column_keys))
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    pid = ncpu
    file_id = [-1]

    for key in remainder_dict:
        set_name, score_name = key
        remainder_data_list = remainder_dict[key]
        if len(remainder_data_list) > 0:
            remainder_data = {}
            for column_key in column_keys:
                remainder_data[column_key] = np.concatenate([x[column_key] for x in remainder_data_list])
            buffer_dict = None
            chunk_save_data(ncpu, pid, file_id, remainder_data, column_keys, out_path, label_str, set_name, buffer_dict,
                          chunk, score_name, id_digit)
            del remainder_data
            gc.collect()
    del remainder_dict
    gc.collect()

    return None


def pad_signal(signal, max_len):
    return np.concatenate([signal, np.zeros(max_len - len(signal), dtype=np.float32)])

def sample_and_save_worker(ncpu, pid, in_file_list, out_path, label_str, set_split_dict,
                              chunk, label, remainder_dict, id_digit, shuffle, read_once, column_keys):
    file_id = [-1]
    data_buffer = {x:[] for x in column_keys}
    buffer_dict = {key:None for key in remainder_dict.keys()}

    for df_idx, df_path in tqdm.tqdm(enumerate(in_file_list), desc=f"Saving {label_str} data", total=len(in_file_list)):
        oom_killer()
        with np.load(df_path) as data:
            for key in column_keys:
                data_buffer[key].append(data[key])

        if len(data_buffer[column_keys[0]]) < read_once and df_idx < len(in_file_list) - 1:
            continue

        else:
            data = {key:np.concatenate(data_buffer[key]) for key in column_keys}

            if shuffle:
                idx = np.random.permutation(len(data[column_keys[0]]))
                for key in column_keys:
                    data[key] = data[key][idx]

            data_buffer = {x:[] for x in column_keys}
            gc.collect()

            score_name = "perfect"
            save_split_data(ncpu, pid, file_id, data, column_keys, out_path, label_str, set_split_dict, chunk, score_name,
                          id_digit, buffer_dict)

            del data
            gc.collect()

    for key in buffer_dict:
        if buffer_dict[key] is not None:
            remainder_dict[key].append(buffer_dict[key])
    del buffer_dict
    gc.collect()

    return None


def save_split_data(ncpu, pid, file_id, data, column_keys, out_path, label_str, set_split_dict, chunk, score_name, id_digit, buffer_dict):

    ## slice for set by index - use cumsum to get the index
    set_idx = np.cumsum([0] + [int(np.floor(len(data[column_keys[0]]) * split_ratio)) for split_ratio in set_split_dict.values()])

    for idx, set_name in enumerate(set_split_dict):
        set_idx_start = set_idx[idx]
        set_idx_end = set_idx[idx+1]
        set_data = {key:data[key][set_idx_start:set_idx_end] for key in column_keys}
        buffer = buffer_dict[(set_name,score_name)]

        if buffer is not None:
            set_data = {key:np.concatenate([buffer[key], set_data[key]]) for key in column_keys}
            buffer_dict[(set_name,score_name)] = None
            gc.collect()

        chunk_save_data(ncpu, pid, file_id, set_data, column_keys, out_path, label_str, set_name, buffer_dict, chunk, score_name, id_digit)

        del set_data
        gc.collect()

    return None


def chunk_save_data(ncpu, pid, file_id, set_data, column_keys, out_path, label_str, set_name, buffer_dict, chunk, score_name, id_digit):

    len_set_data = len(set_data[column_keys[0]])
    for chunk_idx in range(0, len_set_data // chunk + 1):
        file_id[0] += 1
        out_data_id = (ncpu+1) * file_id[0] + pid
        chunk_data = {key:val[chunk_idx * chunk:min((chunk_idx + 1) * chunk, len_set_data)] for key, val in set_data.items()}
        if len(chunk_data[column_keys[0]]) == chunk:
            save_path = f"{out_path}/score-{score_name}/{set_name}/{label_str}/{str(out_data_id).zfill(id_digit)}.npz"
            if os.path.exists(save_path):
                printmessage(f"File {save_path} already exists - overwriting.", msg_type="warning")
            np.savez_compressed(save_path, **chunk_data)
            del chunk_data
            gc.collect()

        elif buffer_dict is not None:
            ## should happen only once per iteration
            buffer = buffer_dict[(set_name,score_name)]
            if buffer is not None:
                buffer_dict[(set_name,score_name)] = {key:np.concatenate([buffer[key], chunk_data[key]]) for key in column_keys}
            else:
                buffer_dict[(set_name,score_name)] = chunk_data.copy()

    return None



def main():
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(0, 1000000)

    os.makedirs(args.out_path, exist_ok=True)

    for set_name in ["train", "val"]:
        # for score_name in ["all","perfect"]:
        for score_name in ["perfect"]:
            for label in ["pos", "neg"]:
                os.makedirs(f"{args.out_path}/score-{score_name}/{set_name}/{label}", exist_ok=True)

    if args.pos_path is not None:
        sample_and_save(args.pos_path, args.out_path, args.cpu, label = 1, chunk = args.chunk)

    if args.neg_path is not None:
        sample_and_save(args.neg_path, args.out_path, args.cpu, label = 0, chunk = args.chunk)

    return None


if __name__ == "__main__":
    main()

