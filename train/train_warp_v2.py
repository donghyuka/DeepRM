import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from train.dataloader_aug_v2 import load_dataset, NanoporeDataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torchmetrics.classification as cm
import argparse
import os
import time
import tqdm
import numpy as np
from utils.utils import printmessage
import importlib
from utils import augmentations_dev as aug
from functools import partial
from utils.interp1d import interp1d

def parse_args():
    parser = argparse.ArgumentParser("Train Transformer Model")
    parser.add_argument("--gpu", type=int, default = 4)
    parser.add_argument("--batch", dest="batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch", dest="eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--data", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/dataset/ver021324/main/")
    parser.add_argument("--output", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/model")
    parser.add_argument("--tb", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/tensorboard")
    parser.add_argument("--model", type=str, default="transformer_prototype_v11")
    parser.add_argument("--es_delta", type=float, default=1e-5)
    parser.add_argument("--es_patience", type=int, default=50)
    parser.add_argument("--es_start", type=int, default=1000)
    parser.add_argument("--disk_shard_size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--enc_dim", type=int, default=512)
    parser.add_argument("--lin_dim", type=int, default=1024)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--enc_layer", type=int, default=6)
    parser.add_argument("--lin_layer", type=int, default=4)
    parser.add_argument("--enc_dropout", type=float, default=0.1)
    parser.add_argument("--lin_dropout", type=float, default=0.2)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--kmer_size", type=int, default=5)
    parser.add_argument("--signal_size", type=int, default=30)
    parser.add_argument("--spectrogram_size", type=int, default=21)
    parser.add_argument("--block_len", type=int, default=17)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--t_act", type=str, default="gelu")
    parser.add_argument("--lin_act", type=str, default="gelu")
    parser.add_argument("--lr_step", type=int, default=1000)
    parser.add_argument("--lr_interval", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--class_ratio", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--profiler", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--read_every", type=int, default=None)
    parser.add_argument("--rlrop", type=float, default=None)
    parser.add_argument("--soft", type=float, default=None)
    parser.add_argument("--loss", type=str, default="BCE")
    parser.add_argument("--score_feature", type=bool, default=False)
    parser.add_argument("--gpu_pool", type=int, nargs="+", default=None)
    parser.add_argument("--cut_overlap", type=bool, default=False)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=512)
    parser.add_argument("--aug", dest="aug_fraction", type=float, default=0.1)
    parser.add_argument("--stride", dest="signal_stride", type=int, default=6)

    strfttime = time.strftime("%Y%m%d-%H%M%S")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size * 4
    if args.name is None:
        args.name = f"AIRNA-{args.model.split('_')[-1]}-{strfttime}"
    if args.read_every is None:
        args.read_every = args.disk_shard_size
    if args.save_interval is None:
        args.save_interval = args.eval_interval
    if args.gpu_pool is None:
        args.gpu_pool = list(range(args.gpu))
    else:
        if len(args.gpu_pool) < args.gpu:
            raise ValueError("GPU Pool should be the same or larger than the number of GPUs to use.")
    return args



class Trainer:
    def __init__(
            self,
            rank: int,
            gpu_id: int,
            model: torch.nn.Module,
            train_loader: NanoporeDataLoader,
            val_loader: NanoporeDataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_func: torch.nn.Module,
            grad_clip: float,
            metric_func_dict: dict,
            checkpoint_path: str,
            tb_path: str,
            es_start: int,
            es_patience: int,
            es_delta: float,
            model_name: str,
            num_gpu: int,
            lr_interval: int,
            eval_interval: int,
            log_interval: int,
            save_interval: int,
            model_config: dict = None,
            soft_label: float = None,
            score_feature: bool = False,
            cut_overlap: bool = False,
            signal_stride: int = 6,
            aug_fraction: float = 0.1,
    ) -> None:

        self.rank = rank
        self.gpu_id = gpu_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.lr_interval = lr_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.metric_func_dict = metric_func_dict
        self.checkpoint_path = checkpoint_path
        self.es_start = es_start
        self.es_patience = es_patience
        self.es_delta = es_delta
        self.continue_training = 1
        self.best_val_loss = np.inf
        self.best_val_loss_epoch = 0
        self.current_val_loss = 0
        self.current_val_metric_dict = {}
        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0
        self.current_batch_loss = 0
        self.current_interval_loss = 0
        self.current_interval_losses = []
        self.current_lr = 0
        self.pbar = None
        self.model_name = model_name
        self.num_gpu = num_gpu
        self.model_config = model_config
        self.devname = f"{os.uname()[1]}-{self.gpu_id}"
        self.tb_path = tb_path
        self.soft_label = soft_label
        self.score_feature = score_feature
        self.cut_overlap = cut_overlap
        self.signal_stride = signal_stride
        self.aug_fraction = aug_fraction
        self.aug_list = self._get_aug_list()
        self.histogram = False
        self.kmer_size = model_config["kmer_size"]
        self.seq_len = model_config["seq_len"]
        self.block_len = model_config["block_len"]
        self.ext_seq_len = self.seq_len + self.kmer_size - 1
        self.unit_size = int(self.ext_seq_len / self.block_len)

        assert self.kmer_size % 2 == 1, f"Kmer size should be odd, but got {self.kmer_size}"
        assert self.ext_seq_len % self.block_len == 0, f"Extended sequence length should be divisible by kmer size, but got {self.ext_seq_len} and {self.block_len}"

        if self.rank == 0:
            self.tb_writer = SummaryWriter(tb_path)
        else:
            self.tb_writer = None

        # self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #                                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=0),
        #                                        on_trace_ready=torch.profiler.tensorboard_trace_handler(self.tb_path, self.devname))

        self.eval_sources, self.eval_targets = self._cache_eval_data()

        self.target_start_idx = (self.block_len // 2) * self.unit_size - (self.kmer_size // 2)
        self.target_end_idx = self.target_start_idx + self.unit_size

        ## END of __init__


    def _get_aug_list(self):
        fraction = self.aug_fraction
        ## MOVING AVERAGE MAGNITUDE WARP
        movmag = partial(aug.moving_magnitude_warp, fraction=fraction, min_sigma=0.1, max_sigma=0.2, n_knots = 40)
        ## WINDOWED TIME WARP
        winwarp = partial(aug.window_warp, fraction=0.5, min_window_ratio = 0.05, max_window_ratio = 0.10,
                          min_window_count = 5, max_window_count = 20, sigma = 0.4)
        ## TIME WARP
        timewarp = partial(aug.time_warp, fraction=0.5, min_sigma=0.2, max_sigma=0.5, n_knots = 30)
        ## GAUSSIAN JITTER
        jitter = partial(aug.jitter, fraction=fraction, min_sigma=0.15, max_sigma=0.3)
        ## SPIKE NOISE
        spike = partial(aug.jitter, fraction=fraction, min_sigma=0.3, max_sigma=1.0, dropout = 0.95)
        ## STEP NOISE
        step = partial(aug.step, fraction=fraction, min_sigma=0.15, max_sigma=0.3, dropout = 0.95)
        ## SLOPE NOISE
        slope = partial(aug.slope, fraction=fraction, magnitude = 0.5,)
        ## DRIFT NOISE
        drift = partial(aug.drift, fraction=fraction, min_sigma=0.15, max_sigma=0.3, n_knots = 10)
        aug_list = [movmag, winwarp, timewarp, jitter, spike, step, drift, slope]
        return aug_list

    def _augment_signal(self, signal, pad_mask, seg_len, shuffle_order = True):

        with torch.no_grad():
            if shuffle_order:
                aug_list = np.random.permutation(self.aug_list)
            else:
                aug_list = self.aug_list

            for i, aug_func in enumerate(aug_list):
                signal, pad_mask, seg_len = aug_func(signal, pad_mask=pad_mask, seg_len=seg_len)


        return signal, pad_mask, seg_len


    def _cache_eval_data(self):
        sources = []
        targets = []
        with torch.no_grad():
            for source, target in self.val_loader:
                sources.append(source)
                targets.append(target)
        return sources, targets


    def _feed_model(self, source, target):
        src_kmer = source["kmer_token"].to(self.gpu_id)
        src_signal = source["signal_token"].to(self.gpu_id)
        src_seg_len = source["segment_len"].to(self.gpu_id)
        target = target.to(torch.float32).to(self.gpu_id)

        with torch.no_grad():
            src_signal = interp1d(torch.arange(src_signal.shape[1], device=src_signal.device).unsqueeze(0).repeat(src_signal.shape[0],1), src_signal, (src_seg_len / self.unit_size).repeat_interleave(self.unit_size * self.signal_stride, dim=1).cumsum(dim=1)).unfold(1, self.signal_stride * self.kmer_size, self.signal_stride)
            src_kmer = src_kmer.repeat_interleave(self.unit_size, dim=1)[:,self.kmer_size//2:-(self.kmer_size//2)]

        output = self.model(src_kmer, src_signal, self.target_start_idx, self.target_end_idx)
        # output = self.model(src_kmer, src_signal)

        return output, target


    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        output, target = self._feed_model(source, target)
        loss = self.loss_func(output, target)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.current_batch_loss = loss.item()
        dist.barrier()
        time.sleep(0.0001*self.gpu_id)
        evaltext = f"LR {self.current_lr:.3E} | T-Loss {self.current_batch_loss:.3E} | V-Loss {self.current_val_loss:.3E} | "
        evaltext += " | ".join([f"{k.upper()} {v:.3E}" for k,v in self.current_val_metric_dict.items() if (k in ["auroc","ap"])])
        self.pbar.update(1)
        self.pbar.set_postfix_str(evaltext)
        return None


    def _run_epoch(self):
        self.train_loader.set_epoch(self.current_epoch)
        self.val_loader.set_epoch(self.current_epoch)
        self.current_batch = 0
        self.model.train()
        self.current_lr = self.optimizer.param_groups[0]['lr']
        colour_choice = ["red", "green", "blue", "yellow", "magenta", "cyan", "white", "black"]
        dist.barrier()
        time.sleep(0.03*self.gpu_id)
        with tqdm.tqdm(total=len(self.train_loader) // self.num_gpu, desc=f"[GPU {self.gpu_id}] Epoch {self.current_epoch}",
                       position=self.rank, colour=colour_choice[self.rank%len(colour_choice)], smoothing = 0) as self.pbar:

            for source, targets in self.train_loader:
                self._run_batch(source, targets)
                self.current_interval_losses.append(self.current_batch_loss)

                if self.current_step % self.log_interval == 0:
                    current_interval_loss = np.mean(self.current_interval_losses)
                    self.current_interval_losses = []
                    ## ALL REDUCE LOSS
                    dist.barrier()
                    current_interval_loss = torch.tensor(current_interval_loss).to(self.gpu_id)
                    dist.all_reduce(current_interval_loss, op=dist.ReduceOp.SUM)
                    self.current_interval_loss = current_interval_loss / self.num_gpu
                    if self.rank == 0:
                        self.tb_writer.add_scalar("Loss", self.current_interval_loss, self.current_step)
                        self.tb_writer.add_scalar("Learning_Rate", self.current_lr, self.current_step)
                    dist.barrier()

                if self.current_step % self.eval_interval == 0 and self.current_step > 0:
                    self._run_eval()

                if self.current_step % self.save_interval == 0 and self.current_step > 0:
                    dist.barrier()
                    if self.rank == 0:
                        if self.histogram:
                            try:
                                for name, parameter in self.model.named_parameters():
                                    self.tb_writer.add_histogram(name, parameter.clone().cpu().data.numpy(), self.current_step)
                            except:
                                pass
                        self._save_checkpoint()
                    dist.barrier()

                if self.current_step % self.lr_interval == 0:
                    if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                        self.scheduler.step(self.current_val_loss)
                    else:
                        self.scheduler.step()
                    self.current_lr = self.optimizer.param_groups[0]['lr']

                self.current_batch += 1
                self.current_step += 1

        return None


    def _run_eval(self):
        self.model.eval()
        val_loss = []
        outputs = []
        with torch.no_grad():
            for source, target in zip(self.eval_sources, self.eval_targets):
                output, target = self._feed_model(source, target)
                loss = self.loss_func(output, target)
                val_loss.append(loss.item())
                outputs.append(output)

        val_loss = np.mean(val_loss)
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(self.eval_targets, dim=0)
        if self.soft_label is not None:
            targets = torch.where(targets > 0.5, torch.ones_like(targets), torch.zeros_like(targets))
        targets = targets.to(torch.long).to(self.gpu_id)
        metric_dict = {}

        for metric_name, metric_func in self.metric_func_dict.items():
            metric_dict[metric_name] = metric_func(outputs, targets)

        ## ALL REDUCE LOSS and METRICS
        dist.barrier()
        val_loss = torch.tensor(val_loss).to(self.gpu_id)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss = val_loss / self.num_gpu
        for key, value in metric_dict.items():
            metric = value.clone().detach().to(self.gpu_id)
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            metric_dict[key] = metric / self.num_gpu
        self.current_val_loss = val_loss
        self.current_val_metric_dict = metric_dict

        if self.rank == 0:
            self.tb_writer.add_scalar("Val_Loss", val_loss, self.current_step)
            for key, value in metric_dict.items():
                self.tb_writer.add_scalar(f"Val_{key}", value, self.current_step)
        dist.barrier()

        self.model.train()
        return None


    def _save_checkpoint(self):

        if self.best_val_loss - self.current_val_loss > self.es_delta:
            ## Save Model if Improved
            self.best_val_loss = self.current_val_loss
            self.best_val_loss_epoch = self.current_epoch
            torch.save({'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': self.current_val_loss,
                        'metric_dict': self.current_val_metric_dict,
                        'model_config': self.model_config,
                        }, f"{self.checkpoint_path}/{self.model_name}-{self.current_epoch}-{self.current_step}.pt")
            self.continue_training = 1

        elif self.current_epoch > self.es_start and self.current_epoch - self.best_val_loss_epoch > self.es_patience:
            self.continue_training = 0

        else:
            self.continue_training = 1

        return None


    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            dist.barrier()
            self.current_epoch = epoch
            self._run_epoch()
            if self.continue_training == 0:
                printmessage(f"Early Stopping at Epoch {self.current_epoch}")
                break
        if self.rank == 0:
            self.tb_writer.flush()
        return None

    ## END of Class NanoporeTrainer


def setup_ddp(rank,world_size,gpu_id):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu_id)
    return None


def prepare_dataloader(data_path, batch_size, eval_batch_size, disk_shard_size, rank, num_replicas, buffer_size,
                       read_every, seed, class_ratio, prefetch, pin_memory, soft_label, num_workers, signal_stride,
                       kmer_size):

    if rank == 0:
        printmessage(f"Total number of dataloader workers: {num_workers * num_replicas}")

    batch_size = batch_size
    train_pos_data_path = f"{data_path}/train/pos"
    train_neg_data_path = f"{data_path}/train/neg"
    val_pos_data_path = f"{data_path}/val/pos"
    val_neg_data_path = f"{data_path}/val/neg"

    train_loader = load_dataset(train_pos_data_path, train_neg_data_path, batch_size, disk_shard_size, rank, num_replicas,
                                buffer_size, read_every, seed = seed, shuffle = True, drop_last = True, class_ratio = class_ratio,
                                prefetch_factor = prefetch, pin_memory = pin_memory, soft_label=soft_label, num_workers = num_workers,
                                signal_stride=signal_stride, kmer_size=kmer_size)
    val_loader = load_dataset(val_pos_data_path, val_neg_data_path, eval_batch_size, disk_shard_size, rank, num_replicas,
                              buffer_size, read_every, seed = seed, shuffle = False, drop_last = True, class_ratio = class_ratio,
                              prefetch_factor = prefetch, pin_memory = pin_memory, soft_label=soft_label, num_workers = num_workers,
                              signal_stride=signal_stride, kmer_size=kmer_size)

    return train_loader, val_loader


def main_worker(rank, args_dict):
    gpu_id = args_dict["gpu_pool"][rank]
    setup_ddp(rank, args_dict["gpu"],gpu_id)
    TransformerModel = importlib.import_module(f"model.{args_dict['model']}").TransformerModel
    model = TransformerModel(d_model = args_dict["enc_dim"], n_heads = args_dict["head"], d_ff = args_dict["lin_dim"],
                             n_layers = args_dict["enc_layer"], lin_depth = args_dict["lin_layer"],
                             t_act = args_dict["t_act"], lin_act = args_dict["lin_act"],
                             encoder_dropout = args_dict["enc_dropout"], lin_dropout = args_dict["lin_dropout"],
                             kmer_size = args_dict["kmer_size"], signal_size = args_dict["signal_size"],
                             spectrogram_size = args_dict["spectrogram_size"], block_len = args_dict["block_len"], seq_len = args_dict["seq_len"])
    if rank == 0:
        total_params = 0
        for name, parameter in model.named_parameters():
            params = parameter.numel()
            total_params += params
        printmessage(f"Total Params: {total_params:,}")

    model = model.to(gpu_id)

    if args_dict["load_checkpoint"] is not None:
        save_dict = torch.load(args_dict["load_checkpoint"], map_location={'cuda:0': f'cuda:{gpu_id}'})
        model.load_state_dict(state_dict=save_dict["model_state_dict"])
    else:
        save_dict = {}

    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr = args_dict["lr"], weight_decay = args_dict["weight_decay"])

    if args_dict["load_checkpoint"] is not None:
        optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        # ## overwrite lr and weight_decay
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = args_dict["lr"]
        #     param_group['weight_decay'] = args_dict["weight_decay"]


    if args_dict["rlrop"] is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = args_dict["lr_step"],
                                                               threshold = args_dict["rlrop"], threshold_mode = "rel", cooldown = 0,
                                                               min_lr = 1e-6, eps = 1e-8)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args_dict["lr_step"], eta_min = 1e-6)

    if args_dict["load_checkpoint"] is not None:
        scheduler.load_state_dict(save_dict["scheduler_state_dict"])
        save_dict.clear()


    if args_dict["loss"] == "MSE":
        loss_func = torch.nn.MSELoss()
    elif args_dict["loss"] == "BCE":
        loss_func = torch.nn.BCELoss()
    elif args_dict["loss"] == "BCEWL":
        loss_func = torch.nn.BCEWithLogitsLoss()
    elif args_dict["loss"] == "CE":
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss Function {args_dict['loss']} Not Implemented.")

    metric_func_dict = {"acc": cm.BinaryAccuracy().to(gpu_id),
                        "auroc": cm.BinaryAUROC().to(gpu_id),
                        "ap": cm.BinaryAveragePrecision().to(gpu_id),
                        "f-1": cm.BinaryF1Score().to(gpu_id),}

    train_loader, val_loader = prepare_dataloader(args_dict["data"], args_dict["batch_size"], args_dict["eval_batch_size"],
                                                  args_dict["disk_shard_size"], rank, args_dict["gpu"], args_dict["buffer_size"],
                                                  args_dict["read_every"], args_dict["seed"], args_dict["class_ratio"], args_dict["prefetch"],
                                                  pin_memory = args_dict["pin_memory"], soft_label = args_dict["soft"], num_workers = args_dict["workers"],
                                                  signal_stride = args_dict["signal_stride"], kmer_size = args_dict["kmer_size"])
    trainer = Trainer(rank, gpu_id, model, train_loader, val_loader, optimizer, scheduler, loss_func, args_dict["grad_clip"], metric_func_dict,
                      args_dict["output"], args_dict["tb"], args_dict["es_start"], args_dict["es_patience"],
                      args_dict["es_delta"], args_dict["name"], args_dict["gpu"], args_dict["lr_interval"], args_dict["eval_interval"],
                      args_dict["log_interval"], args_dict["save_interval"], model_config = args_dict,
                      soft_label = args_dict["soft"], score_feature = args_dict["score_feature"], cut_overlap = args_dict["cut_overlap"],
                      signal_stride = args_dict["signal_stride"], aug_fraction = args_dict["aug_fraction"])
    printmessage(f"[GPU {gpu_id}] Trainer Setup Complete.")
    trainer.train(args_dict["epochs"])
    printmessage(f"[GPU {gpu_id}] Training Loop Complete.")
    dist.destroy_process_group()
    return None


def main_master():
    args = parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.makedirs(os.path.join(args.output, args.name), exist_ok=True)
    os.makedirs(os.path.join(args.tb, args.name), exist_ok=True)
    args.output = os.path.join(args.output, args.name)
    args.tb = os.path.join(args.tb, args.name)
    if args.seed is None:
        args.seed = np.random.randint(0, 10000000)
    printmessage("Training Program Started.")
    printmessage(f"Seed: {args.seed}")
    printmessage(f"Using {args.gpu} GPUs.")
    args_dict = vars(args)
    mp.spawn(main_worker, nprocs=args.gpu, args=(args_dict,))
    printmessage(f"Training Program Complete.")
    return None


if __name__ == "__main__":
    main_master()

