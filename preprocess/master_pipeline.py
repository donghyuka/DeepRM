

import os, argparse
from utils.utils import printmessage


def restructure_directory(args):
    ## Check if pod5 files exist
    pod5_list = os.listdir(args.input)
    pod5_list = [i for i in pod5_list if i.endswith(".pod5")]

    if len(pod5_list) == 0:
        printmessage(f"No pod5 files found in {args.input}. Attempting directory restructuring.")

        if args.rename == "":
            raise ValueError(f"No pod5 files found in {args.input}. Please specify --rename argument")

        rename_path = os.path.join(os.path.dirname(args.input), args.rename)
        os.rename(args.input, rename_path)
        args.input = rename_path

        ## check if input directory has a single subdirectory
        subdir_list = os.listdir(args.input)
        if len(subdir_list) != 1:
            raise ValueError(f"Input directory {args.input} contains multiple subdirectories.")

        subdir_path = os.path.join(args.input, subdir_list[0])
        rename_path = os.path.join(args.input, args.rename)
        os.rename(subdir_path, rename_path)
        args.input = rename_path

        ## check if input directory has "pod5_pass", "pod5_fail", "pod5_skip" subdirectories
        subdir_list = os.listdir(args.input)
        if not any(x in subdir_list for x in ["pod5_pass", "pod5_fail", "pod5_skip"]):
            if "pod5" in subdir_list:
                pass
            else:
                raise ValueError(f"Input directory {args.input} does not contain pod5_pass and pod5_fail subdirectories.")

        else:
            os.makedirs(f"{args.input}/pod5", exist_ok=True)

            ## move pod5 files to pod5 directory
            for subdir in ["pod5_pass", "pod5_fail", "pod5_skip"]:
                subdir_path = os.path.join(args.input, subdir)
                if os.path.exists(subdir_path):
                    cmd = f"mv {subdir_path}/*.pod5 {args.input}/pod5/"
                    os.system(cmd)

            ## check if empty directories remain
            for subdir in ["pod5_pass", "pod5_fail", "pod5_skip"]:
                subdir_path = os.path.join(args.input, subdir)
                if os.path.exists(subdir_path):
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
    parser.add_argument("--dorado", "-d", type=str, default="/extdata3/baeklab/Hyeonseo/bin/dorado-0.7.2", help="Dorado path")
    parser.add_argument("--cpu", "-t", type=int, default=120, help="Number of threads")
    parser.add_argument("--gpu", "-g", type=str, default="cuda:all", help="GPU device")
    parser.add_argument("--batch", "-b", type=int, default=None, help="Dorado Batch size")
    parser.add_argument("--qcut", "-q", type=int, default=7, help="Dorado BQ cutoff")
    parser.add_argument("--rename", "-r", type=str, default="", help="rename input directory")
    parser.add_argument("--dag_cfg", "-c", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/res/config/dag/240202_87BB.json", help="DAG config file")
    parser.add_argument("--step", "-s", type=int, nargs="+", default=[1,], help="Step to run")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input directory {args.input} does not exist")
    if args.step == 0:
        args.step = [1,2,3,4]
    if not any(x in args.step for x in [1,2,3,4]):
        raise ValueError(f"Invalid step argument: {args.step}")
    return args

def main():
    args = parse_args()

    printmessage(f"Input directory: {args.input}")
    printmessage(f"Output directory: {args.output}")
    restructure_directory(args)
    os.makedirs(args.output, exist_ok=True)

    wdir = f"{args.output}/intermediates/"
    bam_path = f"{wdir}/dorado_output.bam"
    pod5_path = args.input
    block_df_path = f"{wdir}/block_df.pkl"
    signal_path = f"{wdir}/segmented_tokenized/"
    dorado_model_path = f"{args.dorado}/model/rna004_130bps_sup@v5.0.0"
    toml_path = "/extdata3/baeklab/Hyeonseo/bin/dorado-0.4.3/model/rna004_130bps_sup@v3.0.1/config.toml"
    # signal_block_path = f"{signal_path}/block"
    # token_path = f"{args.output}/token"
    os.makedirs(wdir, exist_ok=True)

    if 1 in args.step:
        ## step 1. Run Dorado Basecaller and SAMtools
        printmessage(f"[Step 1/3] Running Dorado Basecaller and SAMtools")
        if args.batch is not None:
            args.batch = f"-b {args.batch}"
        else:
            args.batch = ""
        cmd = f"{args.dorado}/bin/dorado basecaller--chunksize 12000 -x {args.gpu} {args.batch} --min-qscore 0 --emit-moves --estimate-poly-a {dorado_model_path} {pod5_path} > {bam_path}"
        printmessage(cmd)
        os.system(cmd)
        cmd = f"samtools sort -@ {args.cpu} -o {bam_path} {bam_path}"
        printmessage(cmd)
        os.system(cmd)
        cmd = f"samtools index -@ {args.cpu} {bam_path}"
        printmessage(cmd)
        os.system(cmd)

    if 2 in args.step:
        ## step 2. Run dag_extract_cb.py
        printmessage(f"[Step 2/3] Running DAG-based CB Extraction")
        cmd = f"python -m preprocess.dag_extract_cb --cpu {args.cpu} --input {bam_path} --output {block_df_path} --rbq {args.qcut} --cfg {args.dag_cfg}"
        printmessage(cmd)
        os.system(cmd)

    if 3 in args.step:
        ## Step 3. Run segment_normalize_signal.py
        printmessage(f"[Step 3/3] Running Signal Segmentation, Normalization, and FFT")
        cmd = f"python -m preprocess.segment_normalize_signal --skip_intermediate --cpu {args.cpu} --pod5 {pod5_path} --bam {bam_path} --block {block_df_path} --output {signal_path} --toml {toml_path}"
        printmessage(cmd)
        os.system(cmd)

    pass

if __name__ == '__main__':
    main()
