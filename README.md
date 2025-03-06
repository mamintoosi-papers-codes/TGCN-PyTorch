# T-GCN-PyTorch


This is a PyTorch implementation of T-GCN in the following paper: [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320).

A stable version of this repository can be found at [the official repository](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch).

Note that [the original implementation](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-TensorFlow) is in TensorFlow, which performs a tiny bit better than this implementation for now.

## Requirements

* numpy
* pandas
* torch
* torchmetrics>=0.11

## Model Training

* YAML config file

    ```bash
    # GCN
    python main.py fit --config configs/gcn.yaml
    # GRU
    python main.py fit --config configs/gru.yaml
    # T-GCN
    python main.py fit --config configs/tgcn.yaml
    ```

---

### GSL (Graph Structure Learning) 

This program incorporates Graph Structure Learning (GSL) for estimating the adjacency matrix. The adjacency matrix is essential for training and inference. Here's how the program handles the adjacency matrix:

1. **Loading an Existing Matrix**:  
   If the estimated adjacency matrix file (e.g., `W_est_losloop_pre_len1.npy`) exists in the `data/` folder, the program will load it directly for computations. This ensures efficiency by utilizing precomputed matrices.

2. **Estimating a New Matrix**:  
   If the required adjacency matrix file cannot be found in the `data/` folder, the program will estimate it using GSL. Once computed, the new adjacency matrix will be saved for future use. The program provides a log to indicate this process:

   Example output:
   ```
   100%|██████████| 180000/180000.0 [10:03<00:00, 298.33it/s]  
   [2025-03-05 13:39:45,765 INFO]Using device: cuda
   File data/W_est_losloop_pre_len1.npy did not exist.  The adjacency matrix estimated by GSL is computed and saved.
   ```

This dual functionality ensures flexibility and reusability of precomputed results while accommodating dynamic estimation when necessary. 

---
