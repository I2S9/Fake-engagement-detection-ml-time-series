"""
Sequence preparation for sequential models (LSTM, TCN).

This module provides functions to transform time series data into fixed-length
sequences suitable for deep learning models.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_sequences(
    df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    feature_columns: Optional[List[str]] = None,
    label_column: str = "label",
    seq_len: int = 48,
    stride: int = 1,
    min_seq_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create fixed-length sequences from time series data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data
    id_column : str
        Name of the ID column
    timestamp_column : str
        Name of the timestamp column
    feature_columns : list of str, optional
        Columns to use as features. If None, uses all numeric columns except id and label
    label_column : str
        Name of the label column
    seq_len : int
        Length of each sequence (number of time steps)
    stride : int
        Step size for creating sequences (1 = no overlap, >1 = overlapping)
    min_seq_len : int, optional
        Minimum sequence length required. If None, uses seq_len

    Returns
    -------
    tuple
        X (sequences), y (labels), video_ids (list of video IDs for each sequence)
    """
    if feature_columns is None:
        # auto-detect feature columns
        feature_columns = [
            col
            for col in df.columns
            if col not in [id_column, timestamp_column, label_column]
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    if min_seq_len is None:
        min_seq_len = seq_len

    sequences = []
    labels = []
    video_ids = []

    # group by ID
    for video_id, group in df.groupby(id_column):
        group = group.sort_values(timestamp_column).reset_index(drop=True)

        if len(group) < min_seq_len:
            # skip videos that are too short
            continue

        # extract features and label
        X_group = group[feature_columns].values
        y_group = group[label_column].iloc[0]  # same label for all sequences from same video

        # convert label to binary
        y_binary = 1 if y_group == "fake" else 0

        # create sequences with sliding window
        for i in range(0, len(X_group) - seq_len + 1, stride):
            sequence = X_group[i : i + seq_len]
            sequences.append(sequence)
            labels.append(y_binary)
            video_ids.append(video_id)

    X = np.array(sequences)
    y = np.array(labels)
    video_ids = np.array(video_ids)

    return X, y, video_ids


def normalize_sequences(
    X: np.ndarray,
    method: str = "standardize",
    per_series: bool = False,
    scaler: Optional[object] = None,
) -> Tuple[np.ndarray, object]:
    """
    Normalize sequences.

    Parameters
    ----------
    X : np.ndarray
        Input sequences [n_sequences, seq_len, n_features]
    method : str
        Normalization method: 'standardize' (z-score) or 'minmax' (0-1)
    per_series : bool
        If True, normalize each sequence independently. If False, normalize globally
    scaler : object, optional
        Pre-fitted scaler to use. If None, fits a new scaler

    Returns
    -------
    tuple
        Normalized sequences, fitted scaler
    """
    if method not in ["standardize", "minmax"]:
        raise ValueError("method must be 'standardize' or 'minmax'")

    if per_series:
        # normalize each sequence independently
        X_normalized = np.zeros_like(X)
        for i in range(X.shape[0]):
            sequence = X[i]
            if method == "standardize":
                scaler_seq = StandardScaler()
            else:
                scaler_seq = MinMaxScaler()
            X_normalized[i] = scaler_seq.fit_transform(sequence)
        return X_normalized, None
    else:
        # normalize globally across all sequences
        n_sequences, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)

        if scaler is None:
            if method == "standardize":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            scaler.fit(X_reshaped)

        X_normalized_reshaped = scaler.transform(X_reshaped)
        X_normalized = X_normalized_reshaped.reshape(n_sequences, seq_len, n_features)

        return X_normalized, scaler


def prepare_sequences_for_training(
    df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    feature_columns: Optional[List[str]] = None,
    label_column: str = "label",
    seq_len: int = 48,
    stride: int = 1,
    normalize: bool = True,
    normalization_method: str = "standardize",
    normalize_per_series: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Complete pipeline to prepare sequences for training.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data
    id_column : str
        Name of the ID column
    timestamp_column : str
        Name of the timestamp column
    feature_columns : list of str, optional
        Columns to use as features
    label_column : str
        Name of the label column
    seq_len : int
        Length of each sequence
    stride : int
        Step size for creating sequences
    normalize : bool
        Whether to normalize sequences
    normalization_method : str
        Normalization method: 'standardize' or 'minmax'
    normalize_per_series : bool
        Whether to normalize each series independently
    test_size : float
        Proportion of data for testing
    val_size : float
        Proportion of data for validation (from training set)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test,
        video_ids_train, video_ids_val, video_ids_test, scaler, feature_names
    """
    from sklearn.model_selection import train_test_split

    # create sequences
    X, y, video_ids = create_sequences(
        df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        feature_columns=feature_columns,
        label_column=label_column,
        seq_len=seq_len,
        stride=stride,
    )

    # normalize if requested
    scaler = None
    if normalize:
        X, scaler = normalize_sequences(
            X, method=normalization_method, per_series=normalize_per_series
        )

    # split into train/val/test
    # first split: train+val vs test
    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        X, y, video_ids, test_size=test_size, random_state=random_state, stratify=y
    )

    # second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train_val,
        y_train_val,
        ids_train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_train_val,
    )

    # get feature names
    if feature_columns is None:
        feature_columns = [
            col
            for col in df.columns
            if col not in [id_column, timestamp_column, label_column]
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "video_ids_train": ids_train,
        "video_ids_val": ids_val,
        "video_ids_test": ids_test,
        "scaler": scaler,
        "feature_names": feature_columns,
    }

