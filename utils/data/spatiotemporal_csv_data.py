import numpy as np
import torch
from utils.data.functions import (
    load_features,
    load_adjacency_matrix,
    generate_torch_datasets,
)

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
}

class SpatioTemporalCSVData:
    """
    A simple class (not a LightningDataModule) that loads
    spatio-temporal data from CSV files and produces train/val datasets.
    """

    def __init__(
        self,
        dataset_name: str,
        seq_len: int = 12,
        pre_len: int = 3,
        split_ratio: float = 0.8,
        normalize: bool = True,
        **kwargs
    ):
        self.dataset_name = dataset_name
        self._feat_path = DATA_PATHS[self.dataset_name]["feat"]
        self._adj_path = DATA_PATHS[self.dataset_name]["adj"]
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize

        self._feat = load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        self._adj = load_adjacency_matrix(self._adj_path)

    def get_datasets(self):
        # returns train_dataset, val_dataset
        train_dataset, val_dataset = generate_torch_datasets(
            self._feat,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )
        return train_dataset, val_dataset

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj

    @property
    def num_nodes(self):
        return self._adj.shape[0]


# If you want to preserve the same "LightningDataModule" style name, define:
class SpatioTemporalCSVDataModule:
    """
    Deprecated: for reference only. This is the old Lightning approach.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 32,
        seq_len: int = 12,
        pre_len: int = 3,
        split_ratio: float = 0.8,
        normalize: bool = True,
        **kwargs
    ):
        self.dataset_name = dataset_name
        self._feat_path = DATA_PATHS[self.dataset_name]["feat"]
        self._adj_path = DATA_PATHS[self.dataset_name]["adj"]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._feat = load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        self._adj = load_adjacency_matrix(self._adj_path)

    def setup(self, stage: str = None):
        self.train_dataset, self.val_dataset = generate_torch_datasets(
            self._feat,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj

    @property
    def num_nodes(self):
        return self._adj.shape[0]
