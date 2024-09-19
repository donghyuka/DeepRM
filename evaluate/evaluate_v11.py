import torch
import os, glob
import argparse
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from evaluate.inference_dataloader_npz_v2 import load_dataset
from utils.utils import printmessage
import torch.multiprocessing as mp
import tqdm
import importlib

## 1. Load Eval Data and Model
## 2. Run Inference.
## 3. Create Site-level Predictions. There is no need to use PILEUP when evaluating on a sampled dataset.
## 4. Evaluate against ground truth labels and save evaluation results.
## 5. Plot evaluation results.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, nargs="+", help="Model path")
    parser.add_argument("--data", "-d", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/runs/exp_MRNA/ON0090/ON0090/result/block/block", help="Data path")
    parser.add_argument("--output", "-o", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/inference/", help="Output path")
    parser.add_argument("--batch", "-b", type=int, default=10000, help="Batch size")
    parser.add_argument("--shard", "-s", type=int, default=10000, help="Shard size")
    parser.add_argument("--gpu", "-g", type=int, default=4, help="GPU device")
    parser.add_argument("--nfile", "-n", type=int, default=16, help="Number of files to load")
    parser.add_argument("--prefetch", "-p", type=int, default=16, help="Number of files to load")
    parser.add_argument("--worker", "-w", type=int, default=8, help="Number of workers per GPU")
    parser.add_argument("--postfix", "-x", type=str, default="", help="Postfix for output directory")
    parser.add_argument("--flush", "-f", type=int, default=100, help="Flush interval for intermediate results.")
    parser.add_argument("--no_bq", action="store_true", help="No BQ")
    parser.add_argument("--motor_only", action="store_true", help="Motor Only")
    parser.add_argument("--no_dwell", action="store_true", help="No Dwell")
    parser.add_argument("--resume", action="store_true", help="Resume terminated inference.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ## Subdirectories for results: inference, pileup, evaluation, plot
    inference_path = f"{args.output}/inference"
    plot_path = f"{args.output}/plot"
    os.makedirs(inference_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    run_inference(args)
    return None


def setup_ddp(rank,world_size):
    if world_size == 0:
        ## use CPU.
        return None

    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    return None

def run_inference(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    args_dict = vars(args)
    printmessage("Inference Program Started.")
    if args_dict["gpu"] > 0:
        printmessage(f"Using {args.gpu} GPUs.")
    else:
        printmessage("Using CPU.")

    model_list = []
    for model_path in args_dict["model"]:
        if model_path.endswith(".pt"):
            model_list.append(model_path)
        elif os.path.isdir(model_path):
            model_list += [x for x in glob.glob(f"{model_path}/*.pt")]
        else:
            raise ValueError("Invalid model path. It should be a .pt file or a directory containing .pt files.")

    if args_dict["data"].endswith("/"):
        args_dict["data"] = args_dict["data"][:-1]
    if not os.path.isdir(args_dict["data"]):
        raise ValueError("Invalid data path. It should be a directory containing data files.")
    for model in model_list:
        printmessage(f"Running inference: {model}")
        args_dict_model = args_dict.copy()
        args_dict_model["model"] = model
        out_dir = f"{args_dict['output']}/inference/{model.split('/')[-1].split('.')[0]}-{args_dict['data'].split('/')[-1]}"
        if len(args_dict["postfix"]) > 0:
            out_dir = f"{out_dir}-{args_dict['postfix']}"
        print(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        args_dict_model["out_dir"] = out_dir
        mp.spawn(inference_worker, nprocs=max(1,args.gpu), args=(args_dict_model,))
    return None


def inference_worker(rank, args_dict):

    setup_ddp(rank, args_dict["gpu"])
    if args_dict["gpu"] > 0:
        save_dict = torch.load(args_dict["model"], map_location={'cuda:0': f'cuda:{rank}'}, weights_only=False)
    else:
        save_dict = torch.load(args_dict["model"], map_location='cpu', weights_only=False)
    model_config = save_dict["model_config"]


    TransformerModel = importlib.import_module(f"model.{model_config['model']}").TransformerModel
    model = TransformerModel(d_model = model_config["enc_dim"], n_heads = model_config["head"], d_ff = model_config["lin_dim"],
                             n_layers = model_config["enc_layer"], lin_depth = model_config["lin_layer"],
                             t_act = model_config["t_act"], lin_act = model_config["lin_act"],
                             encoder_dropout = model_config["enc_dropout"], lin_dropout = model_config["lin_dropout"],
                             kmer_size = model_config["kmer_size"], signal_size = model_config["signal_size"],
                             spectrogram_size = model_config["spectrogram_size"], block_len = model_config["block_len"],
                             seq_len = model_config["seq_len"], signal_stride = model_config["signal_stride"])
    if rank == 0:
        total_params = 0
        for name, parameter in model.named_parameters():
            params = parameter.numel()
            total_params += params
        printmessage(f"Total Params: {total_params:,}")
    if args_dict["gpu"] > 0:
        model.to(rank)
    model.load_state_dict(state_dict=save_dict["model_state_dict"])
    save_dict.clear()
    if args_dict["gpu"] > 0:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    model.eval()

    if args_dict["resume"]:
        saved = glob.glob(f"{args_dict['out_dir']}/inference_{rank}_*.pkl")
        if len(saved) > 0:
            saved = [int(x.split("/")[-1].split("_")[-1].split(".")[0]) for x in saved]
            saved = max(saved)
        else:
            saved = 0
    else:
        saved = 0

    data_loader = load_dataset(args_dict["data"], args_dict["batch"], args_dict["shard"], rank, max(1,args_dict["gpu"]),
                               num_files_read_once = args_dict["nfile"], prefetch_factor = args_dict["prefetch"],
                               worker = args_dict["worker"],
                               cb_len = model_config["block_len"] + model_config["kmer_size"] - 1,
                               kmer_len = model_config["kmer_size"],
                               sampling = int(model_config["signal_size"] / model_config["kmer_size"]),
                               sig_window = model_config["kmer_size"],
                               resume_from = saved)

    id_list = []
    pred_list = []

    tensor_keys = ["kmer_token", "signal_token", "segment_len", "dwell_motor_token", "dwell_pore_token", "bq_token"]
    data_buffer = {key: None for key in tensor_keys}
    data_buffer["label_id"] = None

    for idx, data in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), smoothing = 0):
        for key in tensor_keys:
            if data_buffer[key] is not None:
                data[key] = torch.cat([data_buffer[key], data[key]], dim=0)
        if data_buffer["label_id"] is not None:
            data["label_id"] = np.concatenate([data_buffer["label_id"], data["label_id"]])

        data_size = len(data["label_id"])

        for batch_id in range(data_size // args_dict["batch"]):
            batch_data = {key: data[key][batch_id * args_dict["batch"]:(batch_id + 1) * args_dict["batch"]] for key in tensor_keys}
            batch_data["label_id"] = data["label_id"][batch_id * args_dict["batch"]:(batch_id + 1) * args_dict["batch"]]
            src_kmer = batch_data["kmer_token"].to(rank)
            src_signal = batch_data["signal_token"].to(rank)
            src_seg_len = batch_data["segment_len"].to(rank)

            if args_dict["no_dwell"]:
                if args_dict["no_bq"]:
                    with torch.no_grad():
                        pred = model(src_kmer, src_signal, src_seg_len)
                else:
                    src_bq = batch_data["bq_token"].to(rank)
                    with torch.no_grad():
                        pred = model(src_kmer, src_signal, src_seg_len, src_bq)

            else:
                src_dwell_motor = batch_data["dwell_motor_token"].to(rank)
                src_bq = batch_data["bq_token"].to(rank)
                if args_dict["motor_only"]:
                    src_dwell_bq = src_dwell_motor
                else:
                    src_dwell_pore = batch_data["dwell_pore_token"].to(rank)
                    if args_dict["no_bq"]:
                        src_dwell_bq = torch.stack([src_dwell_motor, src_dwell_pore], dim=-1)
                    else:
                        src_dwell_bq = torch.stack([src_dwell_motor, src_dwell_pore, src_bq], dim=-1)

                with torch.no_grad():
                    pred = model(src_kmer, src_signal, src_seg_len, src_dwell_bq)

            pred_list.append(pred.cpu().detach().numpy())
            id_list.append(np.array(batch_data["label_id"]))

        ## remaining data to buffer
        if data_size % args_dict["batch"] > 0:
            data_buffer = {key: data[key][(data_size // args_dict["batch"]) * args_dict["batch"]:] for key in tensor_keys}
            data_buffer["label_id"] = data["label_id"][(data_size // args_dict["batch"]) * args_dict["batch"]:]
        else:
            data_buffer = {key: None for key in tensor_keys}
            data_buffer["label_id"] = None

        if idx % args_dict["flush"] == 0 and idx > 0:
            id_list = np.concatenate(id_list)
            pred_list = np.concatenate(pred_list)

            data_df = pd.DataFrame({"label_id": id_list, "pred": pred_list})
            out_path = f"{args_dict['out_dir']}/inference_{rank}_{idx+saved}.pkl"
            data_df.to_pickle(out_path)
            id_list = []
            pred_list = []

    if data_buffer["label_id"] is not None:
        batch_data = data_buffer
        src_kmer = batch_data["kmer_token"].to(rank)
        src_signal = batch_data["signal_token"].to(rank)
        src_seg_len = batch_data["segment_len"].to(rank)
        src_dwell_motor = batch_data["dwell_motor_token"].to(rank)
        src_dwell_pore = batch_data["dwell_pore_token"].to(rank)
        src_bq = batch_data["bq_token"].to(rank)

        if args_dict["motor_only"]:
            src_dwell_bq = src_dwell_motor
        elif args_dict["no_bq"]:
            src_dwell_bq = torch.stack([src_dwell_motor, src_dwell_pore], dim=-1)
        else:
            src_dwell_bq = torch.stack([src_dwell_motor, src_dwell_pore, src_bq], dim=-1)

        with torch.no_grad():
            pred = model(src_kmer, src_signal, src_seg_len, src_dwell_bq)

        pred_list.append(pred.cpu().detach().numpy())
        id_list.append(np.array(batch_data["label_id"]))

    id_list = np.concatenate(id_list)
    pred_list = np.concatenate(pred_list)

    data_df = pd.DataFrame({"label_id": id_list, "pred": pred_list})
    out_path = f"{args_dict['out_dir']}/inference_{rank}_last.pkl"
    data_df.to_pickle(out_path)

    if args_dict["gpu"] > 0:
        dist.destroy_process_group()

    return None



if __name__ == "__main__":
    main()



