import torch
import os, glob
import argparse
import numpy as np
import pandas as pd
import torch.distributed as dist
from evaluate.inference_dataloader_v8 import load_dataset
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
    parser.add_argument("--batch", "-b", type=int, default=40000, help="Batch size")
    parser.add_argument("--shard", "-s", type=int, default=10000, help="Shard size")
    parser.add_argument("--gpu", "-g", type=int, default=4, help="GPU device")
    parser.add_argument("--nfile", "-n", type=int, default=16, help="Number of files to load")
    parser.add_argument("--prefetch", "-p", type=int, default=16, help="Number of files to load")
    parser.add_argument("--worker", "-w", type=int, default=8, help="Number of workers per GPU")
    parser.add_argument("--postfix", "-x", type=str, default="", help="Postfix for output directory")
    parser.add_argument("--no_bq", action="store_true", help="No BQ")
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    ## Subdirectories for results: inference, pileup, evaluation, plot
    inference_path = f"{args.output}/attention"
    plot_path = f"{args.output}/plot"
    os.makedirs(inference_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

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
        out_dir = f"{inference_path}/{model.split('/')[-1].split('.')[0]}-{args_dict['data'].split('/')[-1]}"
        if len(args_dict["postfix"]) > 0:
            out_dir = f"{out_dir}-{args_dict['postfix']}"
        print(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        args_dict_model["out_dir"] = out_dir
        mp.spawn(inference_worker, nprocs=max(1,args.gpu), args=(args_dict_model,))
    return None


def inference_worker(rank, args_dict, flush_interval = 10):

    torch.cuda.set_device(rank)
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
    model.eval()

    data_loader = load_dataset(args_dict["data"], args_dict["batch"], args_dict["shard"], rank, max(1,args_dict["gpu"]),
                               num_files_read_once = args_dict["nfile"], prefetch_factor = args_dict["prefetch"],
                               worker = args_dict["worker"],
                               cb_len = model_config["block_len"] + model_config["kmer_size"] - 1,
                               kmer_len = model_config["kmer_size"],
                               sampling = int(model_config["signal_size"] / model_config["kmer_size"]),
                               sig_window = model_config["kmer_size"])

    id_list = []
    pred_list = []
    block_id_list = []

    attention_keys = []
    torch.backends.mha.set_fastpath_enabled(False)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            attention_keys.append(name)
    attention_dict = {x: [] for x in attention_keys}
    attention_batch_dict = {}

    def hook_fn(module, input, output):
        name = module._get_name()
        attention_batch_dict[name] = output[0].cpu().detach().numpy()
        print(name)
        return None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            module.register_forward_hook(hook_fn)

    for idx, data in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), smoothing = 0):
        if args_dict["gpu"] > 0:
            src_kmer = data["kmer_token"].to(rank)
            src_signal = data["signal_token"].to(rank)
            src_seg_len = data["segment_len"].to(rank)
            if not args_dict["no_bq"]:
                src_bq = data["bq_token"].to(rank)

        else:
            src_kmer = data["kmer_token"]
            src_signal = data["signal_token"]
            src_seg_len = data["segment_len"]
            if not args_dict["no_bq"]:
                src_bq = data["bq_token"]

        with torch.no_grad():
            if not args_dict["no_bq"]:
                pred = model(src_kmer=src_kmer, src_signal=src_signal, src_seg_len=src_seg_len, src_bq=src_bq)
            else:
                pred = model(src_kmer=src_kmer, src_signal=src_signal, src_seg_len=src_seg_len)

        if args_dict["gpu"] > 0:
            pred_list.append(pred.cpu().detach().numpy())
        else:
            pred_list.append(pred.detach().numpy())
        id_list.append(np.array(data["label_id"]))
        block_id_list.append(np.array(data["block_id"]))

        for key in attention_keys:
            attention_dict[key].append(attention_batch_dict[key])
            attention_batch_dict[key] = None

        if idx % flush_interval == 0 and idx > 0:
            id_list = np.concatenate(id_list)
            pred_list = np.concatenate(pred_list)
            block_id_list = np.concatenate(block_id_list)
            print(attention_dict)
            print({k:len(v) for k,v in attention_dict.items()})
            attention_dict = {key: np.concatenate(attention_dict[key]) for key in attention_keys}
            data_df = {"label_id": id_list, "block_id": block_id_list, "pred": pred_list}
            data_df.update(attention_dict)
            data_df = pd.DataFrame(data_df)
            out_path = f"{args_dict['out_dir']}/inference_{rank}_{idx}.pkl"
            data_df.to_pickle(out_path)
            id_list = []
            pred_list = []
            block_id_list = []
            attention_dict = {x: [] for x in attention_keys}

    id_list = np.concatenate(id_list)
    pred_list = np.concatenate(pred_list)
    block_id_list = np.concatenate(block_id_list)
    attention_dict = {key: np.concatenate(attention_dict[key]) for key in attention_keys}
    data_df = {"label_id": id_list, "block_id": block_id_list, "pred": pred_list}
    data_df.update(attention_dict)
    data_df = pd.DataFrame(data_df)
    out_path = f"{args_dict['out_dir']}/inference_{rank}_{idx}.pkl"
    data_df.to_pickle(out_path)

    if args_dict["gpu"] > 0:
        dist.destroy_process_group()

    return None



if __name__ == "__main__":
    main()



