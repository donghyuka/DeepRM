import glob
import numpy as np
import multiprocessing as mp
import h5py
import tqdm
import argparse
import os


def fast5_to_str(fast5_path):

    fast5_data = h5py.File(fast5_path, 'r')
    corr_data = fast5_data['/Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'][()]['base']
    fast5_data.close()
    if len(corr_data) < 1:
        raise ValueError(f"Empty sequence in {fast5_path}")
    event_bases="".join([x.decode() for x in corr_data])

    return event_bases


def worker(fast5_list, output_path):
    seq_list = []
    for fast5 in tqdm.tqdm(fast5_list, desc=f"Extracting"):
        try:
            seq = fast5_to_str(fast5)
            seq_list.append(seq)
        except Exception as e:
            continue
    with open(output_path, 'w') as f:
        f.write("\n".join(seq_list))
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--threads", "-t", type=int, default=120, help="Number of threads")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursive search")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    fast5_list = glob.glob(f"{args.input}/**/*.fast5", recursive=args.recursive)
    print(f"Found {len(fast5_list)} fast5 files")
    fast5_list = np.array_split(fast5_list, args.threads)
    proc_list = []
    for i, fast5_list in enumerate(fast5_list):
        p = mp.Process(target=worker, args=(fast5_list, f"{args.output}/temp.{i}.txt"))
        p.start()
        proc_list.append(p)
    for p in proc_list:
        p.join()
    intermediates = glob.glob(f"{args.output}/temp.*.txt")
    buffer = []
    for intermediate in tqdm.tqdm(intermediates, desc="Merging"):
        with open(intermediate, 'r') as f:
            r = f.read()
            if len(r) > 0:
                buffer.append(r)
    buffer = "\n".join(buffer)
    with open(f"{args.output}/all.txt", 'w') as f:
        f.write(buffer)
    for intermediate in intermediates:
        os.remove(intermediate)

    return None


if __name__ == "__main__":
    main()



