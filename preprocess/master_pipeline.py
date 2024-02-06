import os, argparse
from utils.utils import printmessage


def restructure_directory(args, dir_keyword = "raw"):
    ## Check if pod5 files exist
    pod5_list = os.listdir(args.input)
    pod5_list = [i for i in pod5_list if i.endswith(".pod5")]

    if len(pod5_list) == 0:
        printmessage(f"No pod5 files found in {args.input}. Attempting directory restructuring")

        if args.rename == "":
            raise ValueError(f"No pod5 files found in {args.input}. Please specify --rename argument")

        else:
            printmessage(f"No pod5 files found in {args.input}. Attempting directory restructuring")

        rename_path = os.path.join(os.path.dirname(args.input), args.rename)
        os.rename(args.input, rename_path)
        args.input = rename_path

        ## check if input directory has a single subdirectory
        subdir_list = os.listdir(args.input)
        if len(subdir_list) != 1:
            raise ValueError(f"Input directory {args.input} contains multiple subdirectories.")

        subdir_path = subdir_list[0]
        rename_path = os.path.join(args.input, args.rename)
        os.rename(subdir_path, rename_path)
        args.input = rename_path

        ## check if input directory has a single subdirectory
        subdir_list = os.listdir(args.input)
        if len(subdir_list) != 1:
            raise ValueError(f"Input directory {args.input} contains multiple subdirectories.")

        subdir_path = subdir_list[0]
        rename_path = os.path.join(args.input, dir_keyword)
        os.rename(subdir_path, rename_path)
        args.input = rename_path

        ## check if input directory has "pod5_pass", "pod5_fail" subdirectories
        subdir_list = os.listdir(args.input)
        if "pod5_pass" not in subdir_list or "pod5_fail" not in subdir_list:
            raise ValueError(f"Input directory {args.input} does not contain pod5_pass and pod5_fail subdirectories.")
        os.makedirs(f"{args.input}/pod5", exist_ok=True)

        ## move pod5 files to pod5 directory
        for subdir in ["pod5_pass", "pod5_fail"]:
            subdir_path = os.path.join(args.input, subdir)
            cmd = f"mv {subdir_path}/*.pod5 {args.input}/pod5/"
            os.system(cmd)

        ## check if empty directories remain
        for subdir in ["pod5_pass", "pod5_fail"]:
            subdir_path = os.path.join(args.input, subdir)
            if len(os.listdir(subdir_path)) == 0:
                os.rmdir(subdir_path)

        ## set input directory to pod5 directory
        args.input = f"{args.input}/pod5"
        printmessage(f"Input directory restructured to {args.input}")

    else:
        pass

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    num_cpu = os.cpu_count()
    parser.add_argument("--in", "-i", dest = "input", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--out", "-o", dest = "output", type=str, required=True, help="Output directory")
    parser.add_argument("--dorado", "-d", type=str, default="/extdata3/baeklab/Hyeonseo/bin/dorado-0.4.3", help="Dorado path")
    parser.add_argument("--cpu", "-t", type=int, default=int(num_cpu*0.9), help="Number of threads")
    parser.add_argument("--gpu", "-g", type=str, default="cuda:all", help="GPU device")
    parser.add_argument("--batch", "-b", type=int, default=2560, help="Dorado Batch size")
    parser.add_argument("--qcut", "-q", type=int, default=7, help="Dorado BQ cutoff")
    parser.add_argument("--rename", "-r", type=str, default="", help="rename input directory")
    parser.add_argument("--dag_cfg", "-c", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/res/config/dag/240202_87BB.json", help="DAG config file")
    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input directory {args.input} does not exist")
    os.makedirs(args.output, exist_ok=True)
    return args

def main():
    args = parse_args()

    printmessage(f"Input directory: {args.input}")
    printmessage(f"Output directory: {args.output}")
    restructure_directory(args)

    wdir = f"{args.output}/intermediates/"
    os.makedirs(wdir, exist_ok=True)

    # ## step 1. Run Dorado Basecaller and SAMtools
    # printmessage(f"[Step 1/4] Running Dorado Basecaller and SAMtools")
    dorado_model_path = f"{args.dorado}/model/rna004_130bps_sup@v3.0.1"
    bam_path = f"{wdir}/dorado_output.bam"
    pod5_path = args.input
    # cmd = f"{args.dorado}/bin/dorado basecaller -x {args.gpu} -b {args.batch} --min-qscore 0 --emit-moves --estimate-poly-a {dorado_model_path} {pod5_path} > {bam_path}"
    # printmessage(cmd)
    # os.system(cmd)
    #
    # cmd = f"samtools sort -@ {args.cpu} -o {bam_path} {bam_path}"
    # printmessage(cmd)
    # os.system(cmd)
    # cmd = f"samtools index -@ {args.cpu} {bam_path}"
    # printmessage(cmd)
    # os.system(cmd)

    # ## step 2. Run dag_extract_cb.py
    # printmessage(f"[Step 2/4] Running DAG-based CB Extraction")
    block_df_path = f"{wdir}/block_df.pkl"
    # cmd = f"python -m preprocess.dag_extract_cb --cpu {args.cpu} --input {bam_path} --output {block_df_path} --rbq {args.qcut} --cfg {args.dag_cfg}"
    # printmessage(cmd)
    # os.system(cmd)

    # Step 3. Run segment_normalize_signal.py
    printmessage(f"[Step 3/4] Running Signal Segmentation, Normalization, and FFT")
    signal_path = f"{wdir}/normalized_segment_signal/"
    cmd = f"python -m preprocess.segment_normalize_signal --cpu {args.cpu} --pod5 {pod5_path} --bam {bam_path} --block {block_df_path} --output {signal_path}"
    printmessage(cmd)
    os.system(cmd)

    ## step 4. run tokenizer.py
    printmessage(f"[Step 4/4] Running Tokenization")
    signal_path = f"{signal_path}/block"
    token_path = f"{args.output}/token"
    cmd = f"python -m preprocess.tokenizer --cpu {args.cpu} --signal {signal_path} --output {token_path}"
    printmessage(cmd)
    os.system(cmd)

    pass

if __name__ == '__main__':
    main()
