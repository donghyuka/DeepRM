from utils.utils import printmessage
import argparse
import os
import math



def parse_args():
    parser = argparse.ArgumentParser()
    num_cpu = os.cpu_count()
    parser.add_argument("--in", "-i", dest = "input", type=str, default = None, help="Input BAM PATH")
    parser.add_argument("--out", "-o", dest = "output", type=str, required = True, help="Output directory")
    parser.add_argument("--dorado", "-d", type=str, default="/extdata3/baeklab/Hyeonseo/bin/dorado_0.8.3", help="Dorado path")
    parser.add_argument("--cpu", "-t", type=int, default=int(math.floor(num_cpu * 0.95)), help="Number of threads")
    parser.add_argument("--ref", "-r", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/res/ref/RefSeq_ENSEMBL_111224.fa", help="Reference FASTA")
    parser.add_argument("--sort", "-s", type=str, default="", help="Sort option for samtools")

    args = parser.parse_args()
    if not os.path.exists(args.output):
        raise FileNotFoundError(f"output directory {args.output} does not exist")
    return args

def main():
    args = parse_args()

    wdir = f"{args.output}/reference_test/"
    raw_bam_path = f"{wdir}/dorado_output.realigned.raw.bam"
    bam_path = f"{wdir}/dorado_output.realigned.bam"
    raw_pileup_path = f"{wdir}/dorado_output.pileup.raw.tsv"
    pileup_path = f"{wdir}/dorado_output.pileup.filtered.pkl"
    label_path = f"{args.output}/label/Baeklab.070"
    block_path = f"{args.output}/block/"
    qc_path = f"{args.output}/qc/"

    os.makedirs(wdir, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    ## step 1. Run Dorado Basecaller and SAMtools
    printmessage(f"[Step 1/3] Running Dorado Basecaller and SAMtools")

    # cmd = f"{args.dorado} aligner -t {args.cpu} {args.ref} {args.input} > {raw_bam_path}"
    # printmessage(cmd)
    # os.system(cmd)

    cmd = f"samtools view -@ {args.cpu} -bh -F 276 -o {bam_path} {raw_bam_path}"
    printmessage(cmd)
    os.system(cmd)

    cmd = f"samtools sort -@ {args.cpu} {args.sort} -o {bam_path} {bam_path}"
    printmessage(cmd)
    os.system(cmd)

    cmd = f"samtools index -@ {args.cpu} {bam_path}"
    printmessage(cmd)
    os.system(cmd)

    cmd = f"python -m qc.inspect_alignment -i {bam_path} -o {qc_path} -r {args.ref} -c {args.cpu} -m 30 -b 7"
    printmessage(cmd)
    os.system(cmd)

    cmd = f"python -m qc.inspect_run -i {bam_path} -o {qc_path} --mrna"
    printmessage(cmd)
    os.system(cmd)

    ## step 2. Run Pileup and Label
    printmessage(f"[Step 2/3] Running Pileup and Label")

    cmd = f"samtools mpileup -B -Q 0 -f {args.ref} -a {bam_path} > {raw_pileup_path}"
    printmessage(cmd)
    os.system(cmd)

    cmd = f"python -m utils.filter_pileup -i {raw_pileup_path} -o {pileup_path} -c {args.cpu} -m 1"
    printmessage(cmd)
    os.system(cmd)

    return None


if __name__ == "__main__":
    main()


