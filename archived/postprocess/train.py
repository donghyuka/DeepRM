import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from archived.postprocess.dataloader import load_dataset, NanoporeDataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import argparse
import os
import time
import tqdm
import numpy as np
from utils.utils import printmessage
import importlib
import torchmetrics.classification as cm


def parse_args():
    parser = argparse.ArgumentParser("Train Transformer Model")
    parser.add_argument("--gpu", type=int, default = 4)
    parser.add_argument("--batch", dest="batch_size", type=int, default=128)
    parser.add_argument("--eval_batch", dest="eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--data", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/postprocess/dataset_v2")
    parser.add_argument("--output", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/model")
    parser.add_argument("--tb", type=str, default="/extdata4/baeklab/Hyeonseo/m6A/tensorboard")
    parser.add_argument("--model", type=str, default="mlp_model_v1")
    parser.add_argument("--es_delta", type=float, default=1e-5)
    parser.add_argument("--es_patience", type=int, default=100000)
    parser.add_argument("--es_start", type=int, default=100000)
    parser.add_argument("--disk_shard_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--period", type=int, default=30)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--lr_step", type=int, default=3000)
    parser.add_argument("--lr_interval", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--profiler", type=int, default=0)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--read_every", type=int, default=None)
    parser.add_argument("--rlrop", type=float, default=None)
    parser.add_argument("--loss_ratio", type=int, default=10)
    parser.add_argument("--gpu_pool", type=int, nargs="+", default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--input_dim", type=int, default=70)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout_rate", type=float, default=0.01)
    parser.add_argument("--bag_size", type=int, default=200)

    strfttime = time.strftime("%Y%m%d-%H%M%S")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size * 4
    if args.name is None:
        args.name = f"Postprocess-MLP-{args.model.split('_')[-1]}-{strfttime}"
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
            loss_ratio: int,
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
            bag_size: int,
            model_config: dict = None,
    ) -> None:

        self.rank = rank
        self.gpu_id = gpu_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_func = self._get_mil_loss(loss_ratio, bag_size)
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

        if self.rank == 0:
            self.tb_writer = SummaryWriter(tb_path)
        else:
            self.tb_writer = None

        # self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #                                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=0),
        #                                        on_trace_ready=torch.profiler.tensorboard_trace_handler(self.tb_path, self.devname))

        self.eval_sources, self.eval_targets = self._cache_eval_data()

        ## END of __init__


    def _get_mil_loss(self, loss_ratio, bag_size, epsilon = 1e-6):
        def mil_loss(output, target, mask):
            depth = target[:,2]
            output = output * mask
            output = output.reshape(-1, bag_size)
            assert output.shape[0] == target.shape[0], f"Output and Target should have the same size. Check Bag Size: {bag_size}."

            p_output = torch.clamp(output, 0.0, 1-epsilon)
            p_output = torch.log10(1 - p_output)
            p_output = p_output.sum(dim = 1)
            p_output = p_output / depth
            p_output = 1 - torch.pow(10, p_output)
            p_output = p_output.squeeze()

            dom_output = torch.clamp(output, 0.0, 1.0)
            dom_output = dom_output.sum(dim = 1)
            dom_output = dom_output / depth



            if loss_ratio > 10000:
                loss = torch.nn.functional.mse_loss(dom_output, target[:,1])

            elif loss_ratio < 0.0001:
                loss = torch.nn.functional.binary_cross_entropy(p_output, target[:,0])

            else:
                mse_loss = torch.nn.functional.mse_loss(dom_output, target[:,1])
                bce_loss = torch.nn.functional.binary_cross_entropy(p_output, target[:,0])
                loss = (loss_ratio * mse_loss + bce_loss) / (loss_ratio + 1)

            return p_output, dom_output, loss
        return mil_loss

    def _cache_eval_data(self):
        sources = []
        targets = []
        with torch.no_grad():
            for source, target in self.val_loader:
                sources.append(source)
                targets.append(target)
        return sources, targets

    def _feed_model(self, source, target):
        source = source.to(self.gpu_id)
        mask = source[:, -1:]
        source = source[:, :-1]
        target = target.to(self.gpu_id)
        output = self.model(source)
        return output, target, mask


    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        output, target, mask = self._feed_model(source, target)
        p_output, dom_output, loss = self.loss_func(output, target, mask)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.current_batch_loss = loss.item()
        dist.barrier()
        time.sleep(0.001*self.gpu_id)
        evaltext = f"LR {self.current_lr:.3E} | T-Loss {self.current_batch_loss:.3E} | V-Loss {self.current_val_loss:.3E} | "
        evaltext += " | ".join([f"{k.upper()} {v:.3E}" for k,v in self.current_val_metric_dict.items()])
        self.pbar.update(1)
        self.pbar.set_postfix_str(evaltext)

        if self.rank == 0:
            print

        return None


    def _run_epoch(self):
        self.current_batch = 0
        self.model.train()
        self.current_lr = self.optimizer.param_groups[0]['lr']
        self.train_loader.set_epoch(self.current_epoch)
        self.val_loader.set_epoch(self.current_epoch)
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
        p_outputs = []
        p_targets = []
        with torch.no_grad():
            for source, target in zip(self.eval_sources, self.eval_targets):
                output, target, mask = self._feed_model(source, target)
                p_output, dom_output, loss = self.loss_func(output, target, mask)
                val_loss.append(loss.item())
                p_outputs.append(p_output)
                p_targets.append(target[:,0])


        val_loss = np.mean(val_loss)
        outputs = torch.cat(p_outputs, dim=0)
        targets = torch.cat(p_targets, dim=0)
        targets = targets.to(torch.long).to(self.gpu_id)
        metric_dict = {}

        for metric_name, metric_func in self.metric_func_dict.items():
            metric = metric_func(outputs, targets)
            metric_dict[metric_name] = metric

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


def prepare_dataloader(seed, data_path, batch_size, eval_batch_size, rank, world_size, pin_memory, bag_size):

    batch_size = batch_size
    train_data_path = f"{data_path}/train_balanced.pkl"
    val_data_path = f"{data_path}/val_balanced.pkl"
    num_workers = 1
    prefetch_factor = None

    train_loader = load_dataset(seed, rank, world_size, train_data_path, batch_size, num_workers, pin_memory,
                                True, prefetch_factor, bag_size , shuffle=True)
    val_loader = load_dataset(seed, rank, world_size, val_data_path, eval_batch_size, num_workers, pin_memory,
                              False, prefetch_factor, bag_size, shuffle=False)

    return train_loader, val_loader


def main_worker(rank, args_dict):
    gpu_id = args_dict["gpu_pool"][rank]
    setup_ddp(rank, args_dict["gpu"],gpu_id)
    MLPModel = importlib.import_module(f"postprocess.{args_dict['model']}").MLPModel
    model = MLPModel(input_dim = args_dict["input_dim"],
                     hidden_dim = args_dict["hidden_dim"],
                     output_dim = args_dict["output_dim"],
                     num_layers = args_dict["num_layers"],
                     dropout_rate = args_dict["dropout_rate"])
    if rank == 0:
        total_params = 0
        for name, parameter in model.named_parameters():
            params = parameter.numel()
            total_params += params
        printmessage(f"Total Params: {total_params:,}")

    model = model.to(gpu_id)

    if args_dict["load_checkpoint"] is not None:
        save_dict = torch.load(args_dict["load_checkpoint"], map_location={'cuda:0': f'cuda:{gpu_id}'}, weights_only=False)
        model.load_state_dict(state_dict=save_dict["model_state_dict"])
    else:
        save_dict = {}

    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr = args_dict["lr"], weight_decay = args_dict["weight_decay"])

    if args_dict["load_checkpoint"] is not None:
        optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        ## overwrite lr and weight_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args_dict["lr"]
            param_group['weight_decay'] = args_dict["weight_decay"]


    if args_dict["rlrop"] is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = args_dict["lr_step"],
                                                               threshold = args_dict["rlrop"], threshold_mode = "rel", cooldown = 0,
                                                               min_lr = 1e-6, eps = 1e-8)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args_dict["lr_step"], eta_min = 1e-6)

    if args_dict["load_checkpoint"] is not None:
        # scheduler.load_state_dict(save_dict["scheduler_state_dict"])
        save_dict.clear()

    metric_func_dict = {
                        "auroc": cm.BinaryAUROC().to(gpu_id),
                        "ap": cm.BinaryAveragePrecision().to(gpu_id),}

    train_loader, val_loader = prepare_dataloader(args_dict["seed"], args_dict["data"], args_dict["batch_size"], args_dict["eval_batch_size"],
                                                  rank, args_dict["gpu"], args_dict["pin_memory"], args_dict["bag_size"])
    trainer = Trainer(rank, gpu_id, model, train_loader, val_loader, optimizer, scheduler,  args_dict["loss_ratio"], args_dict["grad_clip"], metric_func_dict,
                      args_dict["output"], args_dict["tb"], args_dict["es_start"], args_dict["es_patience"],
                      args_dict["es_delta"], args_dict["name"], args_dict["gpu"], args_dict["lr_interval"], args_dict["eval_interval"],
                      args_dict["log_interval"], args_dict["save_interval"], args_dict["bag_size"], model_config = args_dict)
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
    printmessage(f"Using {args.gpu} GPUs.")
    args_dict = vars(args)
    if args.seed is None:
        args.seed = np.random.randint(0, 10000000)
    printmessage(f"Seed: {args.seed}")
    mp.spawn(main_worker, nprocs=args.gpu, args=(args_dict,))
    printmessage(f"Training Program Complete.")
    return None


if __name__ == "__main__":
    main_master()

