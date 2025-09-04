"""
Dataloader for Nanopore Dataset from NPZ Files

This module provides an iterator and dataset class for loading
Nanopore data from NPZ files. It supports parallel reading of files
and batching of data for efficient processing in PyTorch.
"""

import glob
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from deeprm.utils import check_deps
from deeprm.utils.logging import get_logger

check_deps.check_torch_available()

import torch
from torch.utils.data import DataLoader, IterableDataset

log = get_logger(__name__)
check_deps.check_torch_available()


class NanoporeDatasetIterator:
    """
    Iterator for loading Nanopore dataset from NPZ files.

    Args:
        file_paths (list): List of file paths to NPZ files.
        cb_len (int): Context block length.
        kmer_len (int): K-mer length.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
    """

    def __init__(self, file_paths, cb_len=21, kmer_len=5, sampling=6, sig_window=5, max_workers=4):

        self.file_paths = file_paths
        self.cb_len = cb_len
        self.kmer_len = kmer_len
        self.sampling = sampling
        self.sig_window = sig_window
        self.cb_lr_pad = (cb_len - kmer_len) // 2
        self.trim = kmer_len // 2
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._file_index = 0
        self._futures = []

    def _read_df(self, path):
        """
        Reads a single NPZ file and returns the data as a dictionary.

        Args:
            path (str): Path to the NPZ file.

        Returns:
            dict: Dictionary containing the data from the NPZ file.
        """
        try:
            npz = np.load(path)
            data = {
                "read_id": npz["read_id"],
                "label_id": npz["label_id"],
                "segment_len": npz["segment_len_arr"],
                "signal_token": npz["signal_token"],
                "kmer_token": npz["kmer_token"],
                "dwell_motor_token": npz["dwell_motor_token"],
                "dwell_pore_token": npz["dwell_pore_token"],
                "bq_token": npz["bq_token"],
            }
            npz.close()
        except Exception:
            log.warning(f"Failed to read {path}, skipping.")
            return None
        return data

    def __iter__(self):
        """
        Returns the iterator object itself.

        Returns:
            NanoporeDatasetIterator: The iterator object.
        """
        return self

    def __next__(self):
        """
        Returns the next data from the iterator.

        Returns:
            dict: Dictionary containing the data from the next NPZ file.

        Raises:
            StopIteration: If there are no more files to read.
        """
        if not self._futures:
            end_index = min(self._file_index + self.max_workers, len(self.file_paths))
            paths_batch = self.file_paths[self._file_index : end_index]
            self._futures = [self.executor.submit(self._read_df, p) for p in paths_batch]
            self._futures = [f for f in self._futures if f is not None]
            self._file_index = end_index

        if not self._futures:
            raise StopIteration

        future = self._futures.pop(0)
        data = future.result()
        if data is None:
            return self.__next__()
        return data


class NanoporeDataset(IterableDataset):
    """
    Iterable dataset for loading Nanopore data from NPZ files.

    Args:
        data_path (str): Path to the directory containing NPZ files.
        rank (int): Rank of the current process.
        num_replicas (int): Number of replicas.
        seed (int): Random seed.
        num_files_read_once (int): Number of files to read at once.
        cb_len (int): Context block length.
        kmer_len (int): K-mer length.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
        resume_from (int): Number of files to skip from the start.
    """

    def __init__(
        self,
        data_path,
        rank,
        num_replicas,
        seed=0,
        num_files_read_once=1000,
        cb_len=21,
        kmer_len=5,
        sampling=6,
        sig_window=5,
        resume_from=0,
    ):

        super().__init__()
        self.data_path = data_path
        self.rank = rank
        self.num_replicas = num_replicas
        self.file_paths = sorted(glob.glob(os.path.join(self.data_path, "*.npz")))
        self.epoch = 0
        self.seed = seed
        self.num_shard = math.ceil(len(self.file_paths) / num_replicas)
        self.num_files_read_once = num_files_read_once
        self.cb_len = cb_len
        self.kmer_len = kmer_len
        self.sampling = sampling
        self.sig_window = sig_window
        self.resume_from = resume_from
        self.skip = 0

    def __iter__(self):
        """
        Returns an iterator for the dataset.

        Returns:
            NanoporeDatasetIterator: Iterator for the dataset.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            id = self.rank
            nw = self.num_replicas
        else:
            id = self.rank * worker_info.num_workers + worker_info.id
            nw = self.num_replicas * worker_info.num_workers

        # Shard file paths across workers
        file_paths = self.file_paths[id::nw]
        if self.resume_from:
            file_paths = file_paths[self.resume_from :]

        return NanoporeDatasetIterator(
            file_paths,
            cb_len=self.cb_len,
            kmer_len=self.kmer_len,
            sampling=self.sampling,
            sig_window=self.sig_window,
            max_workers=16,
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of shards in the dataset.
        """
        return self.num_shard - self.resume_from


class NanoporeDataLoader(DataLoader):
    """
    DataLoader for loading Nanopore data.

    Args:
        dataset (NanoporeDataset): The dataset to load data from.
        num_workers (int): Number of worker processes.
        pin_memory (bool): Whether to pin memory.
        drop_last (bool): Whether to drop the last incomplete batch.
        collate_fn (typing.Callable): Function to collate data into batches.
        prefetch_factor (int): Number of batches to prefetch.
    """

    def __init__(self, dataset: NanoporeDataset, num_workers, pin_memory, drop_last, collate_fn, prefetch_factor):
        shuffle = False
        sampler = None
        batch_size = None
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
        )

    ## END of NanoporeDataLoader


def load_dataset(
    data_path,
    rank,
    num_replicas,
    num_files_read_once=1,
    prefetch_factor=100000,
    worker=16,
    cb_len=21,
    kmer_len=5,
    sampling=6,
    sig_window=5,
    resume_from=0,
):
    """
    Loads the Nanopore dataset using DataLoader.

    Args:
        data_path (str): Path to the directory containing NPZ files.
        batch_size (int): Batch size for loading data.
        rank (int): Rank of the current process.
        num_replicas (int): Number of replicas.
        pad_to (int): Padding length for sequences.
        bq_clip (int): Base quality clipping value.
        num_files_read_once (int): Number of files to read at once.
        prefetch_factor (int): Number of batches to prefetch.
        worker (int): Number of worker processes.
        cb_len (int): Context block length.
        kmer_len (int): K-mer length.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
        resume_from (int): Number of files to skip from the start.

    Returns:
        NanoporeDataLoader: DataLoader for loading the dataset.
    """

    dataset = NanoporeDataset(
        data_path,
        rank,
        num_replicas,
        num_files_read_once=num_files_read_once,
        cb_len=cb_len,
        kmer_len=kmer_len,
        sampling=sampling,
        sig_window=sig_window,
        resume_from=resume_from,
    )
    dataloader = NanoporeDataLoader(
        dataset,
        num_workers=worker,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
    )
    return dataloader


def collate_fn(batch):
    """
    Collate function to process a batch of data from the Nanopore dataset.

    Args:
        batch (list): List of dictionaries containing data from the dataset.

    Returns:
        dict: Dictionary containing processed data ready for model input.
    """
    source = {}
    source["read_id"] = torch.tensor(batch["read_id"])
    source["label_id"] = torch.tensor(batch["label_id"])
    source["segment_len"] = torch.tensor(batch["segment_len"], dtype=torch.int32)
    source["signal_token"] = torch.tensor(batch["signal_token"], dtype=torch.float32)
    source["kmer_token"] = torch.tensor(batch["kmer_token"], dtype=torch.int32)
    source["dwell_bq_token"] = torch.tensor(
        np.stack(
            (
                batch["dwell_motor_token"],
                batch["dwell_pore_token"],
                batch["bq_token"],
            ),
            axis=-1,
        ),
        dtype=torch.float32,
    )
    return source
