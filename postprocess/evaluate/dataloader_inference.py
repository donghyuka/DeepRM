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
        source, target = self.df_to_tensor(self.df_dict[self.data_keys[self.data_indices[self.current_index]]], self.bag_size)
        return source, target

    def df_to_tensor(self, df, pad_to):

        df = df.dropna(axis=0)
        len_df = len(df)

        error_arr = np.stack(df["error"].to_numpy(), axis=0).transpose(0, 2, 1) ## 21, 3
        re_error_arr = np.stack(df["realigned_error"].to_numpy(), axis=0) ## 21, 3
        metadata_arr = df[["pi","flag","mapq","depth","read_bq","block_bq","base_bq",
                           "query_pos", "query_len", "left_soft_clip"]].to_numpy() ## 7 + 3

        pred_arr = df["pred"].to_numpy()
        m6a_level = df["m6a_level"].iloc[0]

        error_arr = torch.tensor(error_arr, dtype=torch.float32)
        re_error_arr = torch.tensor(re_error_arr, dtype=torch.float32)
        metadata_arr = torch.tensor(metadata_arr, dtype=torch.float32)
        pred_arr = torch.tensor(pred_arr, dtype=torch.float32).unsqueeze(1)
        label_tensor = torch.tensor([m6a_level,], dtype=torch.float32)

        label_id = df["label_id"].iloc[0]

        if len_df < pad_to:
            error_arr = torch.cat([error_arr, torch.zeros(pad_to - len_df, 21, 3)], dim = 0)
            re_error_arr = torch.cat([re_error_arr, torch.zeros(pad_to - len_df, 21, 3)], dim = 0)
            metadata_arr = torch.cat([metadata_arr, torch.zeros(pad_to - len_df, 10)], dim = 0)
            pred_arr = torch.cat([pred_arr, torch.zeros(pad_to - len_df, 1)], dim = 0)
            mask_arr = torch.cat([torch.ones((len_df,1), dtype=torch.float32), torch.zeros((pad_to - len_df,1), dtype=torch.float32)], dim = 0)

        elif len_df > pad_to:
            error_arr = error_arr[:pad_to]
            re_error_arr = re_error_arr[:pad_to]
            metadata_arr = metadata_arr[:pad_to]
            pred_arr = pred_arr[:pad_to]
            mask_arr = torch.ones((pad_to,1), dtype=torch.float32)

        else:
            mask_arr = torch.ones((len_df,1), dtype=torch.float32)

        source_tensor = (label_id, error_arr, re_error_arr, metadata_arr, pred_arr, mask_arr)

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
    error_list = []
    re_error_list = []
    metadata_list = []
    pred_list = []
    target_list = []
    mask_list = []
    id_list = []

    for source, target in batch:
        label_id, error, re_error, metadata, pred, mask = source
        error_list.append(error)
        re_error_list.append(re_error)
        metadata_list.append(metadata)
        pred_list.append(pred)
        mask_list.append(mask)
        target_list.append(target)
        id_list.append(label_id)

    error_tensor = torch.stack(error_list, dim = 0)
    re_error_tensor = torch.stack(re_error_list, dim = 0)
    metadata_tensor = torch.stack(metadata_list, dim = 0)
    pred_tensor = torch.stack(pred_list, dim = 0)
    mask_tensor = torch.stack(mask_list, dim = 0)
    target_tensor = torch.stack(target_list, dim = 0)

    source_tensor = (id_list, error_tensor, re_error_tensor, metadata_tensor, pred_tensor, mask_tensor)
    return source_tensor, target_tensor


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


