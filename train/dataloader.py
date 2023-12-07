import torch
from torch.utils.data.dataset import Dataset

class SortishDataset(Dataset):
    def __init__(self, examples, batch_size=16, orderish=True, mega_size=30):
        n = len(examples)
        if orderish:  step = batch_size*mega_size
        else: step = int(batch_size*(1/batch_size))

        self.sortish_examples = []
        for mega_batch in range(0, n, step):
            if mega_batch+step>n:
                self.sortish_examples += sorted(examples.select(range(mega_batch, n)), key=lambda t: len(t['input_ids']))
                break
            self.sortish_examples += sorted(examples.select(range(mega_batch, mega_batch+step)), key=lambda t: len(t['input_ids']))

    def __getitem__(self, i):
        return self.sortish_examples[i]

    def __len__(self):
        return len(self.sortish_examples)
