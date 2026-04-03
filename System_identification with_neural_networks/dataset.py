import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
BASE = os.path.dirname(os.path.abspath(__file__))

def load_data(path=os.path.join(BASE, "system_data.npy"), batch_size=256):
    data = np.load(path)

    ''' Split inputs and targets '''
    X = data[:, :3].astype(np.float32)  # [x, x_dot, F]
    Y = data[:, 3:].astype(np.float32)  # [x_next, x_dot_next]

    # Normalize: zero mean, unit variance
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)

    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std

    ''' Train / Validation split 80/20 '''
    n = len(X_norm)
    split = int(n * 0.8)
    idx = np.random.permutation(n)

    X_train, X_val = X_norm[idx[:split]], X_norm[idx[split:]]
    Y_train, Y_val = Y_norm[idx[:split]], Y_norm[idx[split:]]

    ''' Convert to pytorch tensors '''
    to_t = lambda a: torch.tensor(a)
    train_ds = TensorDataset(to_t(X_train), to_t(Y_train))
    val_ds = TensorDataset(to_t(X_val), to_t(Y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    stats = dict(X_mean=X_mean, X_std=X_std,
                 Y_mean=Y_mean, Y_std=Y_std)
    
    return train_loader, val_loader, stats