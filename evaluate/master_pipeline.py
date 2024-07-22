import os, argparse
from utils.utils import printmessage
import glob
import math


def restructure_directory(args):
    basename = os.path.basename(args.input)
    if not basename == args.name:
        printmessage(f"Renaming {basename} to {args.name}")
        os.rename(args.input, os.path.join(os.path.dirname(args.input), args.name))
        args.input = str(os.path.join(os.path.dirname(args.input), args.name))

    if not os.path.exists(f"{args.input}/{args.name}"):
        if len(os.listdir(args.input)) == 1:
            old_path = os.path.join(args.input, os.listdir(args.input)[0])
            printmessage(f"Moving {old_path} to {args.input}/{args.name}")
            os.rename(old_path, os.path.join(args.input, args.name))
            args.input = str(os.path.join(args.input, args.name))
        else:
            raise ValueError(f"Input directory {args.input} contains more than one directory. Please restructure the input directory manually.")
    else:
        args.input = str(os.path.join(args.input, args.name))

    if 1 in args.step or 3 in args.step:

        if not os.path.exists(f"{args.input}/raw"):
            if len(os.listdir(args.input)) == 1:
                old_path = os.path.join(args.input, os.listdir(args.input)[0])
                printmessage(f"Moving {old_path} to {args.input}/raw")
                os.rename(old_path, os.path.join(args.input, "raw"))
                args.raw_path = str(os.path.join(args.input, "raw"))
            else:
                raise ValueError(f"Input directory {args.input} contains more than one directory. Please restructure the input directory manually.")
        else:
            args.raw_path = str(os.path.join(args.input, "raw"))

        ## Check if there are any pod5 directories
        pod5_dirs = glob.glob(f'{args.raw_path}/pod5*/')
        if len(pod5_dirs) == 0:
            raise FileNotFoundError(f"Input directory {args.raw_path} does not contain any pod5 directories")

        ## merge them all into a single directory
        args.pod5 = os.path.join(args.raw_path, "pod5")

        if not os.path.exists(args.pod5):
            os.makedirs(args.pod5)
            for pod5_dir in pod5_dirs:
                for file in os.listdir(pod5_dir):
                    os.rename(os.path.join(pod5_dir, file), os.path.join(args.pod5, file))
                os.rmdir(pod5_dir)

        else:
            pod5_dirs = [x for x in pod5_dirs if os.path.basename(x[:-1]) != "pod5"]
            for pod5_dir in pod5_dirs:
                for file in os.listdir(pod5_dir):
                    os.rename(os.path.join(pod5_dir, file), os.path.join(args.pod5, file))
                os.rmdir(pod5_dir)

    else:
        args.pod5 = ""
        args.raw_path = ""

    return None


def autoconfig(args):
    if args.input.endswith("/"):
        args.input = args.input[:-1]
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input directory {args.input} does not exist")
    if args.step == 0:
        args.step = [1,2,3]
    if not any(x in args.step for x in [1,2,3]):
        raise ValueError(f"Invalid step argument: {args.step}")
    if args.name is None:
        basename = os.path.basename(args.input)
        if not basename.startswith(args.run_prefix):
            basename = basename.split('[')[-1].split(']')[0]
            if not basename.startswith(args.run_prefix):
                raise ValueError(f"Cannot automatically detect run name. Supply --name option manually.")
            else:
                args.name = basename
        else:
            args.name = basename
    if args.base is None:
        basename = os.path.basename(args.input)
        if not basename.startswith(args.run_prefix):
            try:
                basename = basename.split('-')[1].split('_')[0]
                args.base = basename
            except:
                raise ValueError(f"Cannot automatically detect base of interest. Supply --base option manually.")
        else:
            final_summary_path = glob.glob(f"{args.input}/**/final_summary*.txt", recursive=True)
            if len(final_summary_path) != 1:
                raise ValueError(f"Cannot automatically detect base of interest. Supply --base option manually.")
            else:
                with open(final_summary_path[0], "r") as f:
                    found = False
                    for line in f:
                        if line.startswith("sample_id"):
                            basename = line.strip().split("=")[1]
                            basename = basename.split('-')[1].split('_')[0]
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Cannot automatically detect base of interest. Supply --base option manually.")
                args.base = basename

    args.base = get_canonical_base(args.base)

    restructure_directory(args)
    if args.output is None:
        args.output = f"{args.input}/result"
    os.makedirs(args.output, exist_ok=True)

    canonical_base = get_canonical_base(args.base)
    args.canonical_base = canonical_base

    printmessage(f"POD5 directory: {args.pod5}")
    printmessage(f"Output directory: {args.output}")
    printmessage(f"Run Name: {args.name}")
    printmessage(f"Base of Interest: {args.base}")

    os.makedirs(args.output, exist_ok=True)

    return args



def get_canonical_base(base):
    modification_dict = {"A": ["A", "cA", "m6A", "m1A", "Am", "I"],
                         "C": ["C", "cC", "m5C", "hm5C", "Cm"],
                         "G": ["G", "cG", "m7G", "m1G", "Gm"],
                         "U": ["U", "cU", "m5U", "Um", "pseU"]}

    for k, v in modification_dict.items():
        if base in v:
            return k

    raise ValueError(f"Invalid base argument: {base}")

    return None

def parse_args():
    parser = argparse.ArgumentParser()
    num_cpu = os.cpu_count()
    parser.add_argument("--in", "-i", dest = "input", type=str, required=True, help="Input directory")
    parser.add_argument("--out", "-o", dest = "output", type=str, default = None, help="Output directory")
    parser.add_argument("--dorado", "-d", type=str, default="/extdata3/baeklab/Hyeonseo/bin/dorado-0.7.2", help="Dorado path")
    parser.add_argument("--cpu", "-t", type=int, default=int(math.floor(num_cpu * 0.95)), help="Number of threads")
    parser.add_argument("--gpu", "-g", type=str, default="cuda:all", help="GPU device")
    parser.add_argument("--batch", "-b", type=int, default=None, help="Dorado Batch size")
    parser.add_argument("--qcut", "-q", type=int, default=7, help="Dorado BQ cutoff")
    parser.add_argument("--step", "-s", type=int, nargs="+", default=[1,2,3], help="Step to run")
    parser.add_argument("--ref", "-f", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/res/ref/isoform/hg38_rna_nrnm.fasta", help="Reference path")
    parser.add_argument("--run_prefix", "-p", type=str, default="ON", help="Run Prefix")
    parser.add_argument("--toml", "-m", type=str, default="/extdata3/baeklab/Hyeonseo/bin/dorado-0.4.3/model/rna004_130bps_sup@v3.0.1/config.toml", help="Dorado TOML file")
    parser.add_argument("--base", "-x", type=str, default="A", help="Base of Interest")
    parser.add_argument("--name", "-r", type=str, default=None, help="Run Name")
    args = parser.parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input directory {args.input} does not exist")
    if args.step == 0:
        args.step = [1,2,3]
    if not any(x in args.step for x in [1,2,3]):
        raise ValueError(f"Invalid step argument: {args.step}")
    return args

def main():
    args = parse_args()
    args = autoconfig(args)

    wdir = f"{args.output}/intermediates/"
    raw_bam_path = f"{wdir}/dorado_output.raw.bam"
    bam_path = f"{wdir}/dorado_output.bam"
    raw_pileup_path = f"{wdir}/dorado_output.pileup.raw.tsv"
    pileup_path = f"{wdir}/dorado_output.pileup.filtered.pkl"
    label_path = f"{args.output}/label/Baeklab.070"
    block_path = f"{args.output}/block/"
    qc_path = f"{args.output}/qc/"

    os.makedirs(wdir, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    os.makedirs(block_path, exist_ok=True)

    if 1 in args.step:
        ## step 1. Run Dorado Basecaller and SAMtools
        printmessage(f"[Step 1/3] Running Dorado Basecaller and SAMtools")
        dorado_model_path = f"{args.dorado}/model/rna004_130bps_sup@v5.0.0"
        if args.batch is not None:
            args.batch = f"-b {args.batch}"
        else:
            args.batch = ""

        cmd = f"{args.dorado}/bin/dorado basecaller --reference {args.ref} --modified-bases m6A --chunksize 12000 -x {args.gpu} {args.batch} --min-qscore 0 --emit-moves --estimate-poly-a {dorado_model_path} {args.pod5} > {raw_bam_path}"
        printmessage(cmd)
        os.system(cmd)

        cmd = f"samtools view -@ {args.cpu} -bh -F 276 -o {bam_path} {raw_bam_path}"
        printmessage(cmd)
        os.system(cmd)

        cmd = f"samtools sort -@ {args.cpu} -o {bam_path} {bam_path}"
        printmessage(cmd)
        os.system(cmd)

        cmd = f"samtools index -@ {args.cpu} {bam_path}"
        printmessage(cmd)
        os.system(cmd)

        cmd = f"python -m qc.inspect_alignment -i {bam_path} -o {qc_path} -r {args.ref} -c {args.cpu} -m 30 -b 7"
        printmessage(cmd)
        os.system(cmd)

        cmd = f"python -m qc.inspect_run -i {bam_path} -o {qc_path}"
        printmessage(cmd)
        os.system(cmd)


    if 2 in args.step:
        ## step 2. Run Pileup and Label
        printmessage(f"[Step 2/3] Running Pileup and Label")

        cmd = f"samtools mpileup -B -Q 0 -f {args.ref} -a {bam_path} > {raw_pileup_path}"
        printmessage(cmd)
        os.system(cmd)

        cmd = f"python -m utils.filter_pileup -i {raw_pileup_path} -o {pileup_path} -c {args.cpu}"
        printmessage(cmd)
        os.system(cmd)

        cmd = f"python -m evaluate.make_eval_label_inhouse -d {pileup_path} -o {label_path} -c {args.cpu}"
        printmessage(cmd)
        os.system(cmd)


    if 3 in args.step:
        ## Step 3. Run tokenize_transcript.py
        printmessage(f"[Step 3/3] Tokenize Transcript")
        cmd = f"python -m evaluate.tokenize_transcript --boi {args.base} --toml {args.toml} -q {args.qcut} -p {args.pod5} -b {bam_path} -o {block_path} -l {label_path}.GP3.depth5_None.twm6astrict.drach.tsv -c {args.cpu} -n normalise -x drach"
        printmessage(cmd)
        os.system(cmd)
        cmd = f"python -m evaluate.tokenize_transcript --boi {args.base} --toml {args.toml} -q {args.qcut} -p {args.pod5} -b {bam_path} -o {block_path} -l {label_path}.GP3.depth5_None.twm6astrict.drach.tsv -c {args.cpu} -n standardise -x drach"
        printmessage(cmd)
        os.system(cmd)
        cmd = f"python -m evaluate.tokenize_transcript --boi {args.base} --toml {args.toml} -p {args.pod5} -b {bam_path} -o {block_path} -l {label_path}.GP3.depth5_None.twm6astrict.tsv -c {args.cpu} -n normalise -x all"
        printmessage(cmd)
        os.system(cmd)
        cmd = f"python -m evaluate.tokenize_transcript --boi {args.base} --toml {args.toml} -p {args.pod5} -b {bam_path} -o {block_path} -l {label_path}.GP3.depth5_None.twm6astrict.tsv -c {args.cpu} -n standardise -x all"
        printmessage(cmd)
        os.system(cmd)

    return None

if __name__ == '__main__':
    main()
