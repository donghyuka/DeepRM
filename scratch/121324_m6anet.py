import os
import argparse
from utils.utils import printmessage

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam", "-b", type=str, required=True, help="BAM file")
    parser.add_argument("--slow5", "-s", type=str, required=True, help="Slow5 file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output directory")
    parser.add_argument("--threads", "-t", type=int, default=120, help="Number of threads")
    return parser.parse_args()

def main():
    args = parse_args()

    kmer_model = "/extdata4/baeklab/Hyeonseo/m6A/m6anet/rna004.nucleotide.5mer.model.txt"
    ref_fasta = "/extdata4/baeklab/Hyeonseo/m6A/res/ref/RefSeq_ENSEMBL_111224.fa"
    fastq = args.bam.replace(".bam", ".fastq")
    eventalign = os.path.join(args.output, "eventalign.tsv")
    os.makedirs(args.output, exist_ok=True)
    dataprep = os.path.join(args.output, "dataprep")
    os.makedirs(dataprep, exist_ok=True)
    final_out = os.path.join(args.output, "out")
    os.makedirs(final_out, exist_ok=True)

    # cmd = "samtools fastq -@ {threads} {bam} > {fastq}"
    # cmd = cmd.format(bam=args.bam, fastq=fastq, threads=args.threads)
    # printmessage(cmd)
    # os.system(cmd)
    #
    # cmd = "f5c index -t {threads} --slow5 {slow5} {fastq}"
    # cmd = cmd.format(slow5=args.slow5, fastq=fastq, threads=args.threads)
    # printmessage(cmd)
    # os.system(cmd)

    cmd = "f5c eventalign --disable-cuda=yes --rna -b {bam} -r {fastq} -g {ref_fasta} -o {eventalign}  --kmer-model {kmer_model}  --slow5 {slow5}  --signal-index --scale-events -x hpc-high -t {threads}"
    cmd = cmd.format(bam=args.bam, fastq=fastq, ref_fasta=ref_fasta, eventalign=eventalign, kmer_model=kmer_model, slow5=args.slow5, threads=args.threads)
    printmessage(cmd)
    os.system(cmd)

    cmd = "m6anet dataprep --eventalign {eventalign} --out_dir {dataprep} --n_processes {threads}"
    cmd = cmd.format(eventalign=eventalign, dataprep=dataprep, threads=args.threads)
    printmessage(cmd)
    os.system(cmd)

    cmd = "m6anet inference --input_dir {dataprep} --out_dir {final_out} --pretrained_model HEK293T_RNA004 --n_processes {threads}"
    cmd = cmd.format(dataprep=dataprep, final_out=args.output, threads=args.threads)
    printmessage(cmd)
    os.system(cmd)

    return None

if __name__ == "__main__":
    main()

