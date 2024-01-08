import pysam, argparse
from tqdm import tqdm
import numpy as np
from utils.utils import printmessage, mean_phred

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--in", dest="bam_path", type=str, required=True, help="Input bam file")
    args.add_argument("--out", dest="out_path", type=str, required=True, help="Output directory")
    args.add_argument("--cpu", dest="cpu", type=int, default=int(os.cpu_count()*0.9), help="Number of CPUs")
    args.add_argument("--bq", dest="bq_thres", type=int, default=7, help="Base quality threshold")
    args = args.parse_args()
    return args

def filter_bam(args):
    bamfile = pysam.AlignmentFile(args.bam_path, "rb", threads=args.cpu, check_sq=False)
    out_bam = pysam.AlignmentFile(args.out_path, "wb", template=bamfile)
    for read in tqdm(bamfile):
        mean_bq = mean_phred(np.array(read.query_qualities, dtype=int))
        if mean_bq < args.bq_thres:
            continue
        out_bam.write(read)
    out_bam.close()
    bamfile.close()
    return None

def main():
    args = parse_args()
    filter_bam(args)
    return None

if __name__ == "__main__":
    main()





