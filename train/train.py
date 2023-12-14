import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import load_dataset
from model.transformer_prototype import TransformerModel
import GPUtil
import transformers


## To Include:
# 1. Distributed Data Parallel
# 2. Learning Rate Scheduler (transformers.get_cosine_with_hard_restarts_schedule_with_warmup)
## Default optimizer is AdamW. (Use gradient clipping)

def main():
    pass

if __name__ == "__main__":
    main()

