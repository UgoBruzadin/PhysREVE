"""
Generic EEG dataset classes usable for any task (motor imagery, seizure detection, etc.).

For task-specific data loading (e.g. BCI IV 2a, CHB-MIT seizure),
see physreve/datasets/.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UnlabeledEEGDataset(Dataset):
    """
    Unlabeled EEG dataset for pretraining.

    Applies per-trial z-score normalisation (mean/std over time axis).

    Args:
        X: np.ndarray of shape (n_trials, n_channels, n_times)
    """
    def __init__(self, X: np.ndarray):
        mu  = X.mean(axis=-1, keepdims=True)
        std = X.std( axis=-1, keepdims=True).clip(min=1e-8)
        self.X = torch.tensor((X - mu) / std, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


class LabeledEEGDataset(Dataset):
    """
    Labeled EEG dataset for fine-tuning and evaluation.

    Applies per-trial z-score normalisation (mean/std over time axis).

    Args:
        X: np.ndarray of shape (n_trials, n_channels, n_times)
        y: np.ndarray of shape (n_trials,) with integer class labels
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        mu  = X.mean(axis=-1, keepdims=True)
        std = X.std( axis=-1, keepdims=True).clip(min=1e-8)
        self.X = torch.tensor((X - mu) / std, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_split_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    batch_size: int   = 16,
    seed:       int   = 42,
) -> tuple:
    """
    Split labeled data into train/val/test and return DataLoaders.

    Args:
        X: (n_trials, n_channels, n_times)
        y: (n_trials,)
        train_frac: fraction of data for training
        val_frac:   fraction for validation (remainder goes to test)
        batch_size: DataLoader batch size
        seed:       random seed for reproducible splits

    Returns:
        train_loader, val_loader, test_loader
    """
    n = len(X)
    n_tr = int(train_frac * n)
    n_v  = int(val_frac * n)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    idx_tr, idx_v, idx_te = idx[:n_tr], idx[n_tr:n_tr + n_v], idx[n_tr + n_v:]

    train_ds = LabeledEEGDataset(X[idx_tr], y[idx_tr])
    val_ds   = LabeledEEGDataset(X[idx_v],  y[idx_v])
    test_ds  = LabeledEEGDataset(X[idx_te], y[idx_te])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def make_pretrain_loader(
    X: np.ndarray,
    batch_size: int = 16,
) -> DataLoader:
    """
    DataLoader for unlabeled pretraining data.

    Args:
        X: (n_trials, n_channels, n_times)
        batch_size: DataLoader batch size

    Returns:
        pretrain_loader
    """
    ds = UnlabeledEEGDataset(X)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
