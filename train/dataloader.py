import torch
from torch.utils.data.dataset import Dataset
from preprocess.compile_dataset import define_schema
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader

def read_petastorm_dataset(path, schema, batch_size=16, shuffle=True, num_workers=0):

    with make_batch_reader(path, schema, num_epochs=None, num_workers=num_workers) as reader:
        dataloader = DataLoader(reader, batch_size=batch_size, shuffle=shuffle)
        for batch in dataloader:
            yield batch


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

def load_dataset(path, batch_size=16, orderish=True, mega_size=30):
    schema = define_schema()
    examples = read_petastorm_dataset(path, schema, batch_size=batch_size)
    dataset = SortishDataset(examples, batch_size=batch_size, orderish=orderish, mega_size=mega_size)
    return dataset