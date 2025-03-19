import numpy as np
import torch
import os  

from utils.data.functions import (
    load_features,
    load_adjacency_matrix,
    generate_torch_datasets,
)
from dagma import utils
from dagma.linear import DagmaLinear

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
}

import numpy as np  

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
        use_gsl: int = 0,  
        **kwargs  
    ):  
        self.dataset_name = dataset_name  
        self._feat_path = DATA_PATHS[self.dataset_name]["feat"]  
        self._adj_path = DATA_PATHS[self.dataset_name]["adj"]  
        self.seq_len = seq_len  
        self.pre_len = pre_len  
        self.split_ratio = split_ratio  
        self.normalize = normalize  
        self.use_gsl = use_gsl  

        # Load the features and adjacency matrix  
        self._feat = load_features(self._feat_path)  
        self._feat_max_val = np.max(self._feat)  
        self._adj = load_adjacency_matrix(self._adj_path)  

    def get_datasets(self):  
        # Generate train and validation datasets  
        train_dataset, val_dataset = generate_torch_datasets(  
            self._feat,  
            self.seq_len,  
            self.pre_len,  
            split_ratio=self.split_ratio,  
            normalize=self.normalize,  
        )  
        
        # Convert datasets to numpy arrays and store them  
        # Get the first element (X) from each tuple in the dataset
        self.train_data = np.array([x[0].numpy() for x in train_dataset])  
        self.val_data = np.array([x[0].numpy() for x in val_dataset])  
        
        return train_dataset, val_dataset  


    def compute_adjacency_matrix(self):  
        """  
        Compute an adjacency matrix based on the training data or load it if it already exists.  
        This method should be called after get_datasets() to ensure train_data is available.  
        """  
        if not hasattr(self, 'train_data'):  
            raise AttributeError("train_data is not available. Please call get_datasets() first.")  

        if self.use_gsl > 0:  
            W_est_file_name = f"data/W_est_{self.dataset_name}_pre_len{self.pre_len}.npy"  

            # Check if the file exists  
            if os.path.exists(W_est_file_name):  
                # If the file exists, load it  
                W_est_all = np.load(W_est_file_name)  
                print(f"File {W_est_file_name} found. Loading existing adjacency matrix estimated by GSL from training data.")  
            else:  
                # If the file does not exist, compute and save it  
                num_nodes = self._adj.shape[0]  
                W_est_all = np.zeros((num_nodes, num_nodes, self.pre_len))  
                data = np.array([x[0] for x in self.train_data])
                
                # Print data statistics
                print(f"\nDataset: {self.dataset_name}")
                print(f"Data shape: {data.shape}")
                print(f"Data mean: {np.mean(data):.4f}")
                print(f"Data std: {np.std(data):.4f}")
                print(f"Data min: {np.min(data):.4f}")
                print(f"Data max: {np.max(data):.4f}")

                for i in range(self.pre_len):  
                    X = data[i::self.pre_len]
                    print(f"\nFor i={i}:")
                    print(f"X shape: {X.shape}")
                    print(f"X mean: {np.mean(X):.4f}")
                    print(f"X std: {np.std(X):.4f}")
                    
                    # Try different lambda values for sz dataset
                    lambda1 = 0.001 if self.dataset_name == "shenzhen" else 0.02
                    model = DagmaLinear(loss_type='l2')
                    w_est = model.fit(X, lambda1=lambda1)
                    
                    print(f"Number of non-zero elements in w_est: {np.count_nonzero(w_est)}")
                    print(f"Mean of non-zero elements: {np.mean(w_est[w_est != 0]):.4f}")
                    print(f"Max of non-zero elements: {np.max(np.abs(w_est)):.4f}")
                    
                    W_est_all[:, :, i] = w_est  

                # Save the computed adjacency matrix estimates  
                np.save(W_est_file_name, W_est_all)  
                print(f"\nFile {W_est_file_name} did not exist. The adjacency matrix estimated by GSL is computed and saved.")
                print(f"Total number of non-zero elements: {np.count_nonzero(W_est_all)}")

            if W_est_all.ndim == 2:  
                W_est = W_est_all > 0  
            elif W_est_all.ndim == 3:  
                W_est = np.any(W_est_all > 0, axis=2)  
                print(f"\nAfter reduction:")
                print(f"Number of non-zero elements: {np.count_nonzero(W_est)}")

            if self.use_gsl == 1:  # (GSL Only)  
                adj = np.zeros(W_est.shape, dtype=int)  
                adj[W_est > 0] = 1  
                self._adj = adj  
                print('GSL computed: Only GSL')  
            else:  # (GSL + adj)  
                self._adj[W_est > 0] = 1  
                print('GSL computed: GSL+Adj')     

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
