import argparse
import os
from utils.utils import printmessage


REF_PATH = "/extdata4/baeklab/Hyeonseo/m6A/res/ref/isoform/hg38_rna_nrnm.fasta"


def parse_args():
    parser = argparse.ArgumentParser()
    num_cpu = os.cpu_count()
    parser.add_argument("--in", "-i", dest = "input", type=str, required=True, help="POD5 Input directory")
    parser.add_argument("--out", "-o", dest = "output", type=str, required=True, help="Output directory")
    parser.add_argument("--dorado", "-d", type=str, default="/extdata3/baeklab/Hyeonseo/bin/dorado-0.4.3", help="Dorado path")
    parser.add_argument("--cpu", "-t", type=int, default=int(num_cpu*0.9), help="Number of threads")
    parser.add_argument("--gpu", "-g", type=str, default="cuda:all", help="GPU device")
    parser.add_argument("--batch", "-b", type=int, default=2560, help="Dorado Batch size")
    parser.add_argument("--mapq", type=int, dest="mapq", help="MAPQ cutoff", default=30)
    parser.add_argument("--bq", type=int, dest="bq", help="BQ cutoff", default=7)
    args = parser.parse_args()
    return args

def align_bam(args):

    ## 1. Run Dorado
    printmessage("[Step 1/4] Running Dorado Basecaller")
    dorado_model_path = f"{args.dorado}/model/rna004_130bps_sup@v3.0.1"
    bam_path = f"{wdir}/dorado_output.bam"
    pod5_path = args.input
    cmd = f"{args.dorado}/bin/dorado basecaller -x {args.gpu} -b {args.batch} --min-qscore 0 --emit-moves --estimate-poly-a {dorado_model_path} {pod5_path} > {bam_path}"
    printmessage(cmd)
    os.system(cmd)

    ## 2. Filter, sort, and index.
    printmessage("[Step 2/4] Filtering, Sorting, and Indexing BAM file")
    cmd = f"samtools view -@ {args.cpu} -bhS -F 4095 -q {args.mapq} {aln_sam_path} | samtools sort -@ {args.cpu} -o {aln_bam_path}"
    printmessage(cmd)
    os.system(cmd)

    cmd = f"samtools index -@ {args.cpu} {aln_bam_path}"
    printmessage(cmd)
    os.system(cmd)

    ## 3. Segment Blocks
    printmessage("[Step 3/4] Segmenting Blocks")
    cmd = f"python segment_transcript.py --cpu {args.cpu} --pod5 {args.pod5} --bam {aln_bam_path} --output {args.output}"
    printmessage(cmd)
    os.system(cmd)


    ## 4. Tokenize Blocks
    printmessage("[Step 4/4] Tokenizing Blocks")

    return None



def main():
    args = parse_args()
    aln_bam_path = align_bam(args)
    print(f"Aligned BAM file: {aln_bam_path}")
    segment_signal(aln_bam_path, args)
    return None


if __name__ == "__main__":
    main()
