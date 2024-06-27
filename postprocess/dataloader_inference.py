import torch
import pickle
import pandas as pd
import numpy as np
import math

class NanoporeDatasetIterator:
    def __init__(self, df_dict, rank, world_size, shuffle, epoch, seed, bag_size):
        self.current_index = -1
        self.df_dict = df_dict
        self.data_keys = np.array(list(self.df_dict.keys()))
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch
        self.seed = seed
        self.bag_size = bag_size
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.data_keys = self.data_keys[torch.randperm(len(self.data_keys), generator=g)]
        if self.world_size > 1:
            self.data_keys = self.data_keys[self.rank::self.world_size]
        ## END of NanoporeDatasetIterator

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data_keys)

    def __next__(self):
        self.current_index += 1
        if self.current_index >= len(self.data_keys):
            raise StopIteration
        key = self.data_keys[self.current_index]
        source, target = self.df_to_tensor(self.df_dict[key], self.bag_size)
        return source, target

    def df_to_tensor(self, df, pad_to):
        len_df = len(df)

        bq_arr = df[["read_bq","block_bq","base_bq"]].to_numpy() ## 3
        bq_arr = np.clip(bq_arr, 0, 40) / 40
        error_arr = np.stack(df["error"].to_numpy(), axis=0) ## 63
        ppfm_arr = df[["pred","pi","flag","mapq"]].to_numpy() ## 4
        id_arr = df["label_id"].to_numpy()

        source_tensor = np.concatenate([ppfm_arr, bq_arr, error_arr], axis = 1) ## 70
        source_tensor = torch.tensor(source_tensor, dtype=torch.float32)
        label_tensor = id_arr

        return source_tensor, label_tensor


class NanoporeDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, batch_size, rank, world_size, shuffle, seed, bag_size):
        super(NanoporeDataset).__init__()
        self.data = data
        self.batch_size = batch_size
        self.dataset_size = len(self.data)
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.bag_size = bag_size

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        return NanoporeDatasetIterator(self.data, self.rank, self.world_size, self.shuffle, self.epoch, self.seed, self.bag_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        return None

    ## END of NanoporeDataset


def collate_fn(batch):
    source_list = []
    target_list = []
    for source, target in batch:
        source_list.append(source)
        target_list.append(target)
    source_tensor = torch.cat(source_list, dim = 0)
    target_list = np.concatenate(target_list)
    return source_tensor, target_list



class NanoporeDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset:NanoporeDataset, batch_size, num_workers, pin_memory, drop_last, collate_fn, prefetch_factor):
        shuffle = False
        sampler = None
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                         drop_last=drop_last, shuffle=shuffle, sampler=sampler, collate_fn=collate_fn,
                         prefetch_factor=prefetch_factor)

    def __len__(self):
        return int(math.ceil(len(self.dataset) / self.batch_size)) + 1

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        return None
    ## END of NanoporeDataLoader


def load_dataset(seed, rank, world_size, data_path, batch_size, num_workers, pin_memory, drop_last, prefetch_factor,bag_size, shuffle, ):
    data = pickle.load(open(data_path, "rb"))
    print("Data Count: ", len(data))
    dataset = NanoporeDataset(data, batch_size, rank, world_size, shuffle, seed, bag_size)
    dataloader = NanoporeDataLoader(dataset, batch_size, num_workers, pin_memory, drop_last, collate_fn, prefetch_factor)
    return dataloader


