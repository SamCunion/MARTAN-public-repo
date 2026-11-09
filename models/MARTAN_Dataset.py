#converts TFDS dataset to a format usable with torch
import torch
import os
import tensorflow_datasets as tfds
from torch.utils.data import IterableDataset

class MARTAN_Dataset(IterableDataset):
    
    def __init__(self, type: int, split: str):
        self.name = f"martan__dataset/MARTAN{str(type)}DS"
        self.data_dir = os.getenv("TFDS_DS_PTH")
        self.split = split
        self._worker_ds = None
        info = tfds.builder(self.name).info
        self.len = info.splits[split].num_examples
    
    def _get_worker_split(self):
        worker = torch.utils.data.get_worker_info()
        return tfds.even_splits(self.split, worker.num_workers)[worker.id]
    
    def _load_for_worker(self):
        if (self._worker_ds is None):
            worker_split = self._get_worker_split()
            self.worker_ds = tfds.load(self.name, split=worker_split, shuffle_files=False)
            self.worker_ds.prefetch(1)
        return self.worker_ds
    
    def __iter__(self):
        ds = self._load_for_worker()
        for example in tfds.as_numpy(ds):
            yield example

    def __len__(self):
        return self.len
