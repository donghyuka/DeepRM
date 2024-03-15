import functools

import torch
import math
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data import DataLoader
import pandas as pd
import glob
from utils.utils import printmessage

## Load Nanopore Dataset from Pickled Pandas DataFrame

class BinaryClassDatasetIterator:
    def __init__(self, pos_file_paths, neg_file_paths, buffer_size, read_every = 1000, shuffle = True, class_ratio = 1):

        self.paths = [neg_file_paths, pos_file_paths]
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.current_df_index = [-1,-1]
        self.current_iterator = [None, None]
        self.current_class = 0
        self.avail_class = [0,1]
        self.len_iterator = [0,0]
        self.class_ratio = class_ratio
        self.buffer_df = [None, None]
        self.read_every = read_every

    def __iter__(self):
        return self

    def __next__(self):
        result = self._next()
        source, target = self.nanopore_row_to_tensor(result,self.current_class)
        return source, target

    def _read_initial(self):

        df_list = []

        len_read = 0
        while (len_read < self.buffer_size) and (self.current_df_index[self.current_class] < len(self.paths[self.current_class]) - 1):
            self.current_df_index[self.current_class] += 1
            df = pd.read_pickle(self.paths[self.current_class][self.current_df_index[self.current_class]])
            df = df[["kmer_token", "bq_token", "position_token", "signal_token", "spectrogram_token", "move_token", "target_mask"]]
            df_list.append(df)
            len_read += len(df)

        df = pd.concat(df_list, ignore_index=True)
        del df_list

        if self.shuffle:
            df = df.sample(frac=1)

        self.buffer_df[self.current_class] = df[self.read_every:]
        df = df[:self.read_every]

        zipped = df.itertuples(index=False)

        self.current_iterator[self.current_class] = zipped
        self.len_iterator[self.current_class] = self.read_every

        return None


    def _read_once(self):

        df_list = [self.buffer_df[self.current_class]]

        len_read = 0
        while (len_read < self.read_every) and (self.current_df_index[self.current_class] < len(self.paths[self.current_class]) - 1):
            self.current_df_index[self.current_class] += 1
            df = pd.read_pickle(self.paths[self.current_class][self.current_df_index[self.current_class]])
            df = df[["kmer_token", "bq_token", "position_token", "signal_token", "spectrogram_token", "move_token", "target_mask"]]
            df_list.append(df)
            len_read += len(df)

        df = pd.concat(df_list, ignore_index=True)
        del df_list

        if self.shuffle:
            df = df.sample(frac=1)

        self.buffer_df[self.current_class] = df[self.read_every:]
        df = df[:self.read_every]

        zipped = df.itertuples(index=False)

        self.current_iterator[self.current_class] = zipped
        self.len_iterator[self.current_class] = self.read_every
        return None


    def _consume_buffer(self):
        df = self.buffer_df[self.current_class]
        zipped = df.itertuples(index=False)

        self.current_iterator[self.current_class] = zipped
        self.buffer_df[self.current_class] = None
        self.len_iterator[self.current_class] = len(df)
        return None


    def _get_rand_class(self, class_ratio):
        rand = torch.randint(0, 1+class_ratio, (1,))
        if rand < 1:
            return 1
        else:
            return 0

    def _next(self):
        ## Randomly decide between positive and negative data
        if len(self.avail_class) == 0:
            raise StopIteration

        elif len(self.avail_class) == 1:
            self.current_class = self.avail_class[0]

        else:
            self.current_class = self._get_rand_class(self.class_ratio)

        if self.current_df_index[self.current_class] == -1:
            self._read_initial()

        elif self.len_iterator[self.current_class] == 0:
            if self.current_df_index[self.current_class] < len(self.paths[self.current_class]) - 1:
                self._read_once()
            elif self.buffer_df[self.current_class] is not None:
                self._consume_buffer()

        try:
            result = next(self.current_iterator[self.current_class])
            self.len_iterator[self.current_class] -= 1

        except StopIteration:
            self.avail_class.remove(self.current_class)
            result = self._next()

        return result


    def nanopore_row_to_tensor(self, row, class_idx):
        ## Columns: "block_id", "motif", "block_score", "kmer_token", "bq_token", "position_token", "signal_token",
        ##          "spectrogram_token", "move_token", "target_mask"
        kmer_token = torch.tensor(row[0], dtype=torch.long)
        bq_token = torch.tensor(row[1], dtype=torch.long)
        position_token = torch.tensor(row[2], dtype=torch.long)
        signal_token = torch.tensor(row[3], dtype=torch.float)
        spectrogram_token = torch.tensor(row[4], dtype=torch.float)
        move_token = torch.tensor(row[5], dtype=torch.long)
        target_mask = torch.tensor(row[6], dtype=torch.float)

        return_dict = {"kmer_token": kmer_token, "bq_token": bq_token, "position_token": position_token,
                       "signal_token": signal_token, "spectrogram_token": spectrogram_token, "move_token": move_token,
                       "target_mask": target_mask}

        label = torch.tensor(class_idx, dtype=torch.long)

        return return_dict, label


    ## END of BinaryClassDatasetIterator


class NanoporeDataset(torch.utils.data.IterableDataset):
    def __init__(self, pos_data_path, neg_data_path, batch_size, disk_shard_size, rank, num_replicas, buffer_size,
                 read_every = None, seed = 0, shuffle = True, drop_last = True, class_ratio = 1):
        super(NanoporeDataset).__init__()

        self.pos_file_paths = pos_data_path
        self.neg_file_paths = neg_data_path
        self.batch_size = batch_size
        self.disk_shard_size = disk_shard_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed
        self.buffer_size = buffer_size
        self.class_ratio = class_ratio
        self.read_every = read_every
        if self.read_every is None:
            self.read_every = self.disk_shard_size

        if self.drop_last:

            self.pos_num_shard = math.floor(len(self.pos_file_paths)/num_replicas)
            self.neg_num_shard = math.floor(len(self.neg_file_paths)/num_replicas)
            self.pos_num_shard = min(self.pos_num_shard, self.neg_num_shard // self.class_ratio)
            self.neg_num_shard = self.pos_num_shard * self.class_ratio
            self.pos_total_num_shard = self.pos_num_shard * num_replicas
            self.pos_dataset_size = self.pos_total_num_shard * disk_shard_size
            self.neg_total_num_shard = self.neg_num_shard * num_replicas
            self.neg_dataset_size = self.neg_total_num_shard * disk_shard_size

        else:
            self.pos_num_shard = math.ceil(len(self.pos_file_paths)/num_replicas)
            self.neg_num_shard = math.ceil(len(self.neg_file_paths)/num_replicas)
            self.pos_num_shard = min(self.pos_num_shard, self.neg_num_shard // self.class_ratio)
            self.neg_num_shard = self.pos_num_shard * self.class_ratio
            self.pos_total_num_shard = self.pos_num_shard * num_replicas
            self.pos_dataset_size = self.pos_total_num_shard * disk_shard_size
            self.neg_total_num_shard = self.neg_num_shard * num_replicas
            self.neg_dataset_size = self.neg_total_num_shard * disk_shard_size

        self.dataset_size = self.pos_dataset_size + self.neg_dataset_size


    def __len__(self):

        return self.dataset_size

    def __iter__(self):
        pos_file_paths = self._deterministic_shuffle_and_sample(self.pos_file_paths, self.pos_num_shard, self.pos_total_num_shard)
        neg_file_paths = self._deterministic_shuffle_and_sample(self.neg_file_paths, self.neg_num_shard, self.neg_total_num_shard)
        ## TODO: Use Torch Multiprocessing to parallelize the data shuffling process (do it while the model is training).
        return BinaryClassDatasetIterator(pos_file_paths, neg_file_paths, buffer_size = self.buffer_size, read_every=self.read_every,
                                          shuffle = self.shuffle, class_ratio = self.class_ratio)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        return None

    def _deterministic_shuffle_and_sample(self, data_path_list, num_shard, total_num_shard):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(data_path_list), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(data_path_list)))  # type: ignore[arg-type]

        if len(indices) < total_num_shard:
            # add extra samples to make it evenly divisible
            padding_size = total_num_shard - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        elif len(indices) > total_num_shard:
            # remove tail of data to make it evenly divisible.
            indices = indices[:total_num_shard]

        assert len(indices) == total_num_shard, f"{len(indices)} != {total_num_shard}"

        indices = indices[self.rank:total_num_shard:self.num_replicas][:num_shard]
        subsampled = [data_path_list[i] for i in indices]

        return subsampled


    ## END of NanoporeDataset


class NanoporeDataLoader(DataLoader):
    def __init__(self, dataset:NanoporeDataset, batch_size, num_workers, pin_memory, drop_last, collate_fn, prefetch_factor):
        shuffle = False
        sampler = None
        self.dataset = dataset
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                         drop_last=drop_last, shuffle=shuffle, sampler=sampler, collate_fn=collate_fn,
                         prefetch_factor=prefetch_factor)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.dataset.set_epoch(epoch)
        return None

    ## END of NanoporeDataLoader


def load_dataset(pos_data_path, neg_data_path, batch_size,
                 disk_shard_size, rank, num_replicas, buffer_size, read_every, seed = 0, shuffle = True, drop_last = True,
                 pad_to = 200, bq_clip = 40, class_ratio = 1, prefetch_factor = 512, pin_memory=True):
    pad_collate_func = functools.partial(pad_collate, pad_to = pad_to, bq_clip = bq_clip)
    ## Use DataLoader to load the dataset
    pos_data_paths = glob.glob(f"{pos_data_path}/*.pkl")
    neg_data_paths = glob.glob(f"{neg_data_path}/*.pkl")
    dataset = NanoporeDataset(pos_data_paths, neg_data_paths, batch_size, disk_shard_size, rank, num_replicas, buffer_size,
                              read_every, seed, shuffle, drop_last, class_ratio = class_ratio)
    dataloader = NanoporeDataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=pin_memory, drop_last=drop_last,
                                    collate_fn = pad_collate_func, prefetch_factor=prefetch_factor)

    return dataloader


def pad_collate(batch, pad_to, bq_clip):
    ## Collate function for DataLoader
    ## Based on NanoporeDataset
    ## Transform into Batch First

    label = [item[1] for item in batch]
    target = torch.stack(label, dim=0)

    token_name_list = ["kmer_token", "bq_token", "position_token", "signal_token", "spectrogram_token",
                       "move_token", "target_mask"]

    source = {}

    ## Zero pad the followings: kmer_token, bq_token, position_token, signal_token, spectrogram_token, move_token
    for token_name in token_name_list:
        token = [item[0][token_name] for item in batch]
        token = torch.nn.utils.rnn.pad_sequence(token, batch_first=True, padding_value=0)
        if pad_to is not None:
            if token.shape[1] < pad_to:
                add_shape = list(token.shape)
                add_shape[1] = pad_to - token.shape[1]
                token = torch.cat((token, torch.zeros(add_shape, dtype=token.dtype)), dim=1)
            else:
                token = token[:, :pad_to]

        source[token_name] = token

    ## clip bq
    source["bq_token"] = torch.clamp(source["bq_token"], 0, bq_clip)

    return source, target
