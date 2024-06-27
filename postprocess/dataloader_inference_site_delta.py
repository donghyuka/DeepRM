import torch
import pickle
import pandas as pd
import numpy as np
import math
from utils.utils import printmessage

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
        self.data_indices = np.arange(len(self.data_keys))
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            np.random.seed(self.seed + self.epoch)
            self.data_indices = np.random.permutation(self.data_indices)
        if self.world_size > 1:
            self.data_indices = self.data_indices[self.rank::self.world_size]
        ## END of NanoporeDatasetIterator

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data_indices)

    def __next__(self):
        self.current_index += 1
        if self.current_index >= len(self.data_indices):
            raise StopIteration
        source, target, id = self.df_to_tensor(self.df_dict[self.data_keys[self.data_indices[self.current_index]]], self.bag_size)
        return source, target, id

    def df_to_tensor(self, df, pad_to):
        len_df = len(df)
        df["depth"] = len_df / pad_to

        ## Sort by pred
        df = df.sort_values("pred", ascending=False)

        error_arr = np.stack(df["error"].to_numpy(), axis=0) ## 21, 3
        error_arr = torch.tensor(error_arr, dtype=torch.float32)
        target_dom = df["dom_pred"].iloc[0]
        target_glori =  df["m6a_level"].iloc[0]
        target_delta = target_dom - target_glori
        id = df["label_id"].iloc[0]

        label_tensor = np.array([target_dom,target_glori,target_delta], dtype=np.float32)

        if len_df < pad_to:
            error_arr = torch.cat([error_arr, torch.zeros(pad_to - len_df, 21, 3)], dim = 0)
            mask_arr = torch.cat([torch.ones((len_df,1), dtype=torch.float32), torch.zeros((pad_to - len_df,1), dtype=torch.float32)], dim = 0)

        elif len_df > pad_to:
            error_arr = error_arr[:pad_to]
            mask_arr = torch.ones((pad_to,1), dtype=torch.float32)

        else:
            mask_arr = torch.ones((len_df,1), dtype=torch.float32)


        source_tensor = (error_arr, mask_arr)

        return source_tensor, label_tensor, id


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
    error_list = []
    target_list = []
    mask_list = []
    id_list = []
    for source, target, id in batch:
        error,  mask = source
        error_list.append(error)
        mask_list.append(mask)
        target_list.append(target)
        id_list.append(id)
    error_tensor = torch.stack(error_list, dim = 0)
    mask_tensor = torch.stack(mask_list, dim = 0)
    target_tensor = np.stack(target_list, axis = 0)
    id_tensor = np.array(id_list)

    source_tensor = (error_tensor, mask_tensor)
    return source_tensor, target_tensor, id_tensor



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
    dataset = NanoporeDataset(data, batch_size, rank, world_size, shuffle, seed, bag_size)
    dataloader = NanoporeDataLoader(dataset, batch_size, num_workers, pin_memory, drop_last, collate_fn, prefetch_factor)
    return dataloader


