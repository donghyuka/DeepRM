import os, argparse
from utils.utils import printmessage
import math


def print_run(cmd, dry):
    """
    Print command and run if not dry run.

    Args:
        cmd (str): Command to run.
        dry (bool): Dry run flag.

    Returns:
        None
    """
    printmessage(cmd)
    if not dry:
        os.system(cmd)
    return None


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Raises:
        FileNotFoundError: If the Dorado installation path does not exist.
        FileNotFoundError: If the Dorado model does not exist.
        FileNotFoundError: If the input directory does not exist.
        ValueError: If the step argument is invalid
    """
    parser = argparse.ArgumentParser()
    num_cpu = os.cpu_count()
    parser.add_argument("--in", "-i", dest = "pod5", type=str, required=True, help="Input directory")
    parser.add_argument("--out", "-o", dest = "output", type=str, required=True, help="Output directory")
    parser.add_argument("--dorado", "-d", type=str, default="", required=True, help = "Dorado installation path")
    parser.add_argument("--ref", "-f", type=str, required=True, help="Reference path")
    parser.add_argument("--model", "-m", type=str, default=None, help="Dorado basecalling model path")
    parser.add_argument("--cpu", "-t", type=int, default=int(math.floor(num_cpu * 0.95)), help="Number of threads")
    parser.add_argument("--gpu", "-g", type=str, default="cuda:all", help="GPU device")
    parser.add_argument("--batch", "-b", type=int, default=None, help="Dorado Batch size")
    parser.add_argument("--qcut", "-q", type=int, default=0, help="Dorado BQ cutoff")
    parser.add_argument("--step", "-s", type=int, nargs="+", default=[1,2,3], help="Step to run")
    parser.add_argument("--base", "-x", type=str, default="A", help="Base of Interest")
    parser.add_argument("--comment", "-c", type=str, default="", help="Comment")
    parser.add_argument("--ivt", "-v", action="store_true", default=False, help="Option for IVT sample. Disables trimming.")
    parser.add_argument("--dry", "-y", action="store_true", default=False, help="Dry run")
    args = parser.parse_args()

    if not os.path.exists(args.dorado):
        raise FileNotFoundError(f"Dorado installation path {args.dorado} does not exist.")

    if args.model is None:
        args.model = f"{args.dorado}/model/rna004_130bps_sup@v5.0.0"
        if not os.path.exists(args.model):
            printmessage(f"Dorado model does not exist. Attempting download...", msg_type="warning")
            cmd = f"{args.dorado}/bin/dorado download --model rna004_130bps_sup@v5.0.0 --directory {args.dorado}/model/"
            print_run(cmd, args.dry)

            if not os.path.exists(args.model):
                raise FileNotFoundError(f"Model {args.model} does not exist.")

    if not os.path.exists(args.pod5):
        raise FileNotFoundError(f"Input POD5 directory {args.pod5} does not exist.")

    if not any(x in args.step for x in [1,2,3]):
        raise ValueError(f"Invalid step argument: {args.step}")

    return args


def main():
    """
    Main function to run the preprocessing pipeline.

    Steps:
        1. Run Dorado Basecaller and SAMtools.
        2. Run Pileup and Label.
        3. Run preproce.py.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        ValueError: If the step argument is invalid or if the run name or base of interest cannot be detected.
    """
    args = parse_args()

    wdir = f"{args.output}/intermediates/"
    raw_bam_path = f"{wdir}/dorado_output.raw.bam"
    bam_path = f"{wdir}/dorado_output.bam"
    raw_pileup_path = f"{wdir}/dorado_output.pileup.raw.tsv"
    pileup_path = f"{args.output}/dorado_output.pileup.filtered.pkl"
    block_path = f"{args.output}/block/"
    qc_path = f"{args.output}/qc/"

    if not args.dry:
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(wdir, exist_ok=True)
        os.makedirs(block_path, exist_ok=True)
        os.makedirs(qc_path, exist_ok=True)

    if 1 in args.step:
        ## step 1. Run Dorado Basecaller and SAMtools
        printmessage(f"[Step 1/3] Running Dorado Basecaller and SAMtools")

        if args.batch is not None:
            args.batch = f"-b {args.batch}"
        else:
            args.batch = ""

        trimming = ""
        if args.ivt:
            trimming = "--no-trim"
            ## When auto-trimming is enabled and the 3’ sequence is unconventional, the move table is corrupted.
            ## It is a weird bug in Dorado. To avoid this, we disable trimming for IVT samples, and minimap2 can simply clip it.

        cmd = f"{args.dorado}/bin/dorado basecaller --reference {args.ref} {trimming} --chunksize 12000 -x {args.gpu} {args.batch} --min-qscore 0 --emit-moves {args.model} {args.pod5} > {raw_bam_path}"
        print_run(cmd, args.dry)

        cmd = f"samtools view -@ {args.cpu} -bh -F 276 -o {bam_path} {raw_bam_path}"
        print_run(cmd, args.dry)

        cmd = f"samtools sort -@ {args.cpu} -o {bam_path} {bam_path}"
        print_run(cmd, args.dry)

        cmd = f"samtools index -@ {args.cpu} {bam_path}"
        print_run(cmd, args.dry)

        cmd = f"python -m qc.inspect_alignment -i {bam_path} -o {qc_path} -r {args.ref} -c {args.cpu} -m 30 -b 7"
        print_run(cmd, args.dry)

        cmd = f"python -m qc.inspect_run -i {bam_path} -o {qc_path} --mrna"
        print_run(cmd, args.dry)


    if 2 in args.step:
        ## step 2. Run Pileup and Label
        printmessage(f"[Step 2/3] Running Pileup and Label")

        cmd = f"samtools mpileup -B -Q 0 -f {args.ref} -a {bam_path} > {raw_pileup_path}"
        print_run(cmd, args.dry)

        cmd = f"python -m inference.filter_mpileup -i {raw_pileup_path} -o {pileup_path} -c {args.cpu} -m 1"
        print_run(cmd, args.dry)


    if 3 in args.step:
        ## Step 3. Run tokenize_transcript.py
        printmessage(f"[Step 3/3] Tokenize Transcript")

        cmd = f"python -m inference.tokenize_transcript --boi {args.base} -q {args.qcut} -p {args.pod5} -b {bam_path} -o {block_path} -l {pileup_path} -c {args.cpu} -x {args.comment}"
        print_run(cmd, args.dry)

    return None

if __name__ == '__main__':
    main()