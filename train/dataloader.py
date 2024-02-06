import torch
from torch.utils.data.dataset import Dataset, DataLoader, RandomSampler, SequentialSampler

## Implement Smart Batching
## Load from Pickled Pandas DataFrame


class NanoporeDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data = data
        self.targets = targets
