"""
PyTorch Dataset and DataLoader for time series sequences.

This module provides PyTorch Dataset classes and DataLoader utilities
for sequential models (LSTM, TCN, etc.).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        video_ids: Optional[np.ndarray] = None,
    ):
        """
        Initialize TimeSeriesDataset.

        Parameters
        ----------
        X : np.ndarray
            Input sequences [n_sequences, seq_len, n_features]
        y : np.ndarray
            Labels [n_sequences]
        video_ids : np.ndarray, optional
            Video IDs for each sequence
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.video_ids = video_ids if video_ids is not None else np.arange(len(X))

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a single sequence and its label.

        Parameters
        ----------
        idx : int
            Index of the sequence

        Returns
        -------
        tuple
            (sequence, label, video_id)
        """
        return self.X[idx], self.y[idx], self.video_ids[idx]


class TimeSeriesDatasetSimple(Dataset):
    """Simplified PyTorch Dataset that returns only (X, y)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Initialize TimeSeriesDatasetSimple.

        Parameters
        ----------
        X : np.ndarray
            Input sequences [n_sequences, seq_len, n_features]
        y : np.ndarray
            Labels [n_sequences]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and its label.

        Parameters
        ----------
        idx : int
            Index of the sequence

        Returns
        -------
        tuple
            (sequence, label)
        """
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    include_video_ids: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Parameters
    ----------
    X_train : np.ndarray
        Training sequences [n_train, seq_len, n_features]
    y_train : np.ndarray
        Training labels [n_train]
    X_val : np.ndarray, optional
        Validation sequences
    y_val : np.ndarray, optional
        Validation labels
    X_test : np.ndarray, optional
        Test sequences
    y_test : np.ndarray, optional
        Test labels
    batch_size : int
        Batch size for DataLoaders
    shuffle_train : bool
        Whether to shuffle training data
    num_workers : int
        Number of worker processes for data loading
    pin_memory : bool
        Whether to pin memory (useful for GPU)
    include_video_ids : bool
        Whether to include video IDs in dataset

    Returns
    -------
    dict
        Dictionary with keys: 'train', 'val', 'test' (if provided)
    """
    dataloaders = {}

    # create training dataloader
    if include_video_ids:
        train_dataset = TimeSeriesDataset(X_train, y_train)
    else:
        train_dataset = TimeSeriesDatasetSimple(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataloaders["train"] = train_loader

    # create validation dataloader
    if X_val is not None and y_val is not None:
        if include_video_ids:
            val_dataset = TimeSeriesDataset(X_val, y_val)
        else:
            val_dataset = TimeSeriesDatasetSimple(X_val, y_val)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        dataloaders["val"] = val_loader

    # create test dataloader
    if X_test is not None and y_test is not None:
        if include_video_ids:
            test_dataset = TimeSeriesDataset(X_test, y_test)
        else:
            test_dataset = TimeSeriesDatasetSimple(X_test, y_test)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        dataloaders["test"] = test_loader

    return dataloaders


def create_dataloaders_from_dict(
    data_dict: dict,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    include_video_ids: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders from dictionary returned by prepare_sequences_for_training.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test
    batch_size : int
        Batch size for DataLoaders
    shuffle_train : bool
        Whether to shuffle training data
    num_workers : int
        Number of worker processes
    pin_memory : bool
        Whether to pin memory
    include_video_ids : bool
        Whether to include video IDs

    Returns
    -------
    dict
        Dictionary with DataLoaders
    """
    return create_dataloaders(
        X_train=data_dict["X_train"],
        y_train=data_dict["y_train"],
        X_val=data_dict.get("X_val"),
        y_val=data_dict.get("y_val"),
        X_test=data_dict.get("X_test"),
        y_test=data_dict.get("y_test"),
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        include_video_ids=include_video_ids,
    )

