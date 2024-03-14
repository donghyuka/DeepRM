import torch
import pysam
import os
import argparse
import numpy as np
import pandas as pd

## 1. Load Eval Data and Model
## 2. Run Inference.
## 3. Use Alignment BAM file and PILEUP to generate site-level predictions.
## 4. Evaluate against ground truth labels and save evaluation results.
## 5. Plot evaluation results.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--data", "-d", type=str, required=True, help="Data path")
    parser.add_argument("--label", "-l", type=str, required=True, help="Label path")
    parser.add_argument("--bam", "-a", type=str, required=True, help="BAM file path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--batch", "-b", type=int, default=100, help="Batch size")
    parser.add_argument("--gpu", "-g", type=str, default="cuda:all", help="GPU device")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ## Subdirectories for results: inference, pileup, evaluation, plot
    inference_path = f"{args.output}/inference"
    pileup_path = f"{args.output}/pileup"
    evaluation_path = f"{args.output}/evaluation"
    plot_path = f"{args.output}/plot"
    os.makedirs(inference_path, exist_ok=True)
    os.makedirs(pileup_path, exist_ok=True)
    os.makedirs(evaluation_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    run_inference(args, inference_path)
    run_pileup(args, inference_path, pileup_path)
    run_evaluation(args, pileup_path, evaluation_path)
    run_plot(args, evaluation_path, plot_path)

    return None

def run_inference(args, inference_path):
    ## Load Model
    model = torch.load(args.model)
    model.to(args.gpu)
    model.eval()

    ## Load Data
    data = torch.load(args.data)
    data.to(args.gpu)

    ## Run Inference
    with torch.no_grad():
        pred = model(data)

    ## Save Inference
    torch.save(pred, f"{inference_path}/inference.pkl")

    return None


def run_pileup(args, inference_path, pileup_path, ncpu = 120):
    ## Load Inference
    pred = torch.load(f"{inference_path}/inference.pkl")
    mod_dict = convert_inference_df_to_mod_dict(pred)
    ## Load BAM file
    bam_in = pysam.AlignmentFile(args.bam, "rb")
    bam_out = pysam.AlignmentFile(f"{pileup_path}/pileup.bam", "wb", template=bam_in)

    ## Save Modification information into bam file using standard MM and ML tags.
    ## MM: Modification status (e.g. MM:Z:A+m,5,12,0 => A is modified at position 6, 19, and 20)
    ## ML: Modification probability (e.g. ML:B:C:10,50,45 => divide by 256 to get the probability between 0 and 1)
    ## This is necessary for PILEUP using SAMTOOLS.

    for read in bam_in.fetch():
        mm_string, ml_string = convert_mod_dict_to_sam_tags(mod_dict[read.query_name])
        read.set_tag("MM", value_type="Z", value=mm_string)
        read.set_tag("ML", value_type="B", value=ml_string)
        bam_out.write(read)

    bam_in.close()
    bam_out.close()

    ## Run Pileup
    cmd = f"samtools mpileup -f {args.ref} {pileup_path}/pileup.bam > {pileup_path}/pileup.pileup"
    os.system(cmd)


    return None


def convert_inference_df_to_mod_dict(inference_df):
    ## 1. Group df by "read_id"
    ## 2. For each group, convert "position" and "mod_prob" to "mod_dict"
    ## 3. Return "mod_dict" as a dictionary of read_id -> mod_dict. So it is a Dict(str, Dict(int, float)).

    group_dict = inference_df.groupby("read_id")
    mod_dict = {}
    for read_id, group in group_dict:
        mod_dict[read_id] = group[["position", "mod_prob"]].set_index("position").to_dict()["mod_prob"]

    return mod_dict


def convert_mod_dict_to_sam_tags(mod_dict, mod_type = "A+m", score_cutoff = 0.05):
    ## Convert mod_dict to SAM tags (MM and ML)
    ## Only include positions with mod_prob > score_cutoff.
    pos_list = [0]
    prob_list = []
    for pos, prob in mod_dict.items():
        if prob > score_cutoff:
            pos_list.append(pos)
            prob_list.append(prob)

    ## Generate MM tags.
    mm_list = [x-y for x, y in zip(pos_list[1:], pos_list[:-1])]
    mm_string = f"{mod_type}+{','.join([str(x) for x in mm_list])}"

    ## Generate ML tags.
    ml_list = [int(x*256) for x in prob_list]
    ml_string = f"{','.join([str(x) for x in ml_list])}"

    return mm_string, ml_string