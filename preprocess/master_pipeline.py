import os, argparse
from utils.utils import printmessage

def parse_args():
    parser = argparse.ArgumentParser()
    num_cpu = os.cpu_count()
    parser.add_argument("--input", "-i", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--dorado", "-d", type=str, default="/extdata3/baeklab/Hyeonseo/bin/dorado-0.4.3", help="Dorado path")
    parser.add_argument("--thread", "-t", type=int, default=int(num_cpu*0.9), help="Number of threads")
    parser.add_argument("--gpu", "-g", type=int, default="cuda:all", help="GPU device")
    parser.add_argument("--batch", "-b", type=int, default=2560, help="Dorado Batch size")
    parser.add_argument("--qcut", "-q", type=int, default=7, help="Dorado BQ cutoff")
    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input directory {args.input} does not exist")
    if os.path.exists(args.output):
        raise FileExistsError(f"Output directory {args.output} already exists")
    os.makedirs(args.out)
    return args

def main():
    args = parse_args()
    wdir = f"{args.output}/intermediates/"
    os.makedirs(wdir)

    ## step 1. Run Dorado Basecaller and SAMtools
    printmessage(f"[Step 1/3] Running Dorado Basecaller and SAMtools")
    dorado_model_path = f"{args.dorado}/model/rna004_130bps_sup@v3.0.1"
    bam_path = f"{wdir}/dorado_output.bam"
    pod5_path = args.input
    cmd = f"dorado basecaller -x {args.gpu} -b {args.batch} --min-qscore {args.qcut} --emit-moves --estimate-poly-a {dorado_model_path} {pod5_path} > {bam_path}"
    print(cmd)
    os.system(cmd)
    cmd = f"samtools sort -@ {args.thread} -o {bam_path} {bam_path}"
    print(cmd)
    os.system(cmd)
    cmd = f"samtools index -@ {args.thread} {bam_path}"
    print(cmd)
    os.system(cmd)

    ## step 2. Run dag_extract_cb.py
    printmessage(f"[Step 2/4] Running DAG-based CB Extraction")
    block_df_path = f"{wdir}/block_df.pkl"
    cmd = f"python dag_extract_cb.py --cpu {args.thread} --input {bam_path} --output {block_df_path}"
    print(cmd)
    os.system(cmd)

    ## Step 3. Run segment_normalize_signal.py
    printmessage(f"[Step 3/4] Running Signal Segmentation, Normalization, and FFT")
    signal_path = f"{wdir}/normalized_segment_signal/"
    cmd = f"python segment_normalize_signal.py --cpu {args.thread} --pod5 {pod5_path} --bam {bam_path} --block {block_df_path} --output {signal_path}"
    print(cmd)
    os.system(cmd)

    ## step 4. run tokenizer.py
    printmessage(f"[Step 4/4] Running Tokenization")
    cmd = f"python tokenizer.py --cpu {args.thread} --signal {signal_path} --block {block_df_path} --output {args.output}"
    print(cmd)
    os.system(cmd)

    pass

if __name__ == '__main__':
    main()
