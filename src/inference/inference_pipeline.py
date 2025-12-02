"""
Inference pipeline for fake engagement detection.

This module provides functions to load trained models and make predictions
on new time series data.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Dict, Union, Tuple, List
import os

from src.models.baselines import BaselineModel, create_baseline_model
from src.models.lstm import create_lstm_model
from src.models.tcn import create_tcn_model
from src.models.autoencoder import create_autoencoder_model
from src.data.preprocess import (
    handle_timezone,
    sort_by_id_and_timestamp,
    resample_timeseries,
    handle_missing_values,
)
from src.data.sequence_preparation import create_sequences, normalize_sequences
from src.features.temporal_features import extract_temporal_features


def load_baseline_model(model_path: str) -> BaselineModel:
    """
    Load a saved baseline model.

    Parameters
    ----------
    model_path : str
        Path to the saved model file (.pkl)

    Returns
    -------
    BaselineModel
        Loaded baseline model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # load model dict to get model_name
    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)

    model_name = model_dict.get("model_name")
    if model_name is None:
        raise ValueError("Model file does not contain model_name")

    # create the correct model instance
    model = create_baseline_model(model_name)
    model.load(model_path)
    return model


def load_sequential_model(
    model_path: str,
    model_type: str,
    config: Optional[Dict] = None,
) -> nn.Module:
    """
    Load a saved sequential model (LSTM, TCN, or Autoencoder).

    Parameters
    ----------
    model_path : str
        Path to the saved model checkpoint (.pth)
    model_type : str
        Type of model: 'lstm', 'tcn', or 'autoencoder'
    config : dict, optional
        Configuration dictionary. If None, tries to load from checkpoint

    Returns
    -------
    nn.Module
        Loaded PyTorch model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    # get config from checkpoint or use provided
    if config is None:
        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            raise ValueError(
                "Config must be provided if not in checkpoint. "
                "Please provide model configuration."
            )

    # create model
    model_config = config.get(model_type, {})
    input_size = model_config.get("input_size")
    seq_len = config.get("data", {}).get("seq_len", 48)

    if input_size is None:
        raise ValueError(f"input_size must be specified in config for {model_type}")

    if model_type == "lstm":
        model = create_lstm_model(model_config)
    elif model_type == "tcn":
        model = create_tcn_model(model_config)
    elif model_type == "autoencoder":
        model_config["seq_len"] = seq_len
        model = create_autoencoder_model(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def preprocess_time_series(
    df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    target_timezone: str = "UTC",
    resample_frequency: Optional[str] = "h",
    handle_missing: bool = True,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Preprocess a new time series using the same steps as training.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series DataFrame
    id_column : str
        Name of the ID column
    timestamp_column : str
        Name of the timestamp column
    target_timezone : str
        Target timezone for timestamps
    resample_frequency : str, optional
        Frequency for resampling (e.g., 'h' for hourly)
    handle_missing : bool
        Whether to handle missing values
    feature_columns : list of str, optional
        Feature columns to use. If None, auto-detects numeric columns

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame
    """
    df = df.copy()

    # handle timezone
    df = handle_timezone(df, timestamp_column=timestamp_column, target_timezone=target_timezone)

    # sort by id and timestamp
    df = sort_by_id_and_timestamp(df, id_column=id_column, timestamp_column=timestamp_column)

    # resample if requested
    if resample_frequency is not None:
        df = resample_timeseries(
            df,
            id_column=id_column,
            timestamp_column=timestamp_column,
            frequency=resample_frequency,
            aggregation_method="sum",
        )

    # handle missing values
    if handle_missing:
        df = handle_missing_values(df, method="forward")

    return df


def prepare_for_baseline_model(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    id_column: str = "id",
    expected_feature_names: Optional[List[str]] = None,
    expected_n_features: Optional[int] = None,
) -> np.ndarray:
    """
    Prepare time series data for baseline model prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame
    feature_columns : list of str, optional
        Feature columns to use. If None, extracts temporal features
    id_column : str
        Name of the ID column
    expected_feature_names : list of str, optional
        Expected feature names from training. If provided, ensures output has same features.

    Returns
    -------
    np.ndarray
        Feature array [n_samples, n_features]
    """
    if feature_columns is None:
        # extract temporal features with same parameters as training
        features_df = extract_temporal_features(
            df,
            id_column=id_column,
            window_sizes=[6, 12, 24],
            autocorr_lags=[1, 6, 12, 24],
            aggregate_per_id=True,
        )
        feature_cols = [col for col in features_df.columns if col not in [id_column, "label"]]
        
        # if expected features are provided, align features
        if expected_feature_names is not None:
            # add missing features with zero values
            missing_features = set(expected_feature_names) - set(feature_cols)
            for feat in missing_features:
                features_df[feat] = 0.0
            
            # reorder to match expected order
            feature_cols = [f for f in expected_feature_names if f in features_df.columns]
        
        X = features_df[feature_cols].values
        
        # if expected number of features is provided, ensure correct size
        if expected_n_features is not None and X.shape[1] != expected_n_features:
            if X.shape[1] < expected_n_features:
                # pad with zeros
                padding = np.zeros((X.shape[0], expected_n_features - X.shape[1]))
                X = np.hstack([X, padding])
            elif X.shape[1] > expected_n_features:
                # truncate (shouldn't happen, but just in case)
                X = X[:, :expected_n_features]
    else:
        # use provided features
        if id_column in df.columns:
            df_grouped = df.groupby(id_column).agg("mean").reset_index()
            X = df_grouped[feature_columns].values
        else:
            X = df[feature_columns].values

    return X


def prepare_for_sequential_model(
    df: pd.DataFrame,
    seq_len: int,
    feature_columns: Optional[List[str]] = None,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    scaler: Optional[object] = None,
    normalize: bool = True,
    normalization_method: str = "standardize",
    normalize_per_series: bool = False,
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Prepare time series data for sequential model prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame
    seq_len : int
        Sequence length
    feature_columns : list of str, optional
        Feature columns to use. If None, auto-detects
    id_column : str
        Name of the ID column
    timestamp_column : str
        Name of the timestamp column
    scaler : object, optional
        Pre-fitted scaler for normalization
    normalize : bool
        Whether to normalize
    normalization_method : str
        Normalization method: 'standardize' or 'minmax'
    normalize_per_series : bool
        Whether to normalize per series

    Returns
    -------
    tuple
        (sequences, scaler) where sequences is [n_sequences, seq_len, n_features]
    """
    # create sequences
    X, _, _ = create_sequences(
        df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        feature_columns=feature_columns,
        seq_len=seq_len,
        stride=seq_len,  # no overlap for inference
        min_seq_len=seq_len,
    )

    # normalize if requested
    if normalize:
        X, scaler = normalize_sequences(
            X,
            method=normalization_method,
            per_series=normalize_per_series,
            scaler=scaler,
        )

    return X, scaler


class InferencePipeline:
    """Pipeline for making predictions on new time series data."""

    def __init__(
        self,
        model_path: str,
        model_type: str,
        config: Optional[Dict] = None,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize inference pipeline.

        Parameters
        ----------
        model_path : str
            Path to the saved model
        model_type : str
            Type of model: 'baseline' or one of 'lstm', 'tcn', 'autoencoder'
        config : dict, optional
            Configuration dictionary (required for sequential models)
        threshold : float
            Decision threshold for binary classification
        device : torch.device, optional
            Device for sequential models (default: cpu)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.config = config
        self.threshold = threshold
        self.device = device if device is not None else torch.device("cpu")

        # load model
        if model_type in ["lstm", "tcn", "autoencoder"]:
            self.model = load_sequential_model(model_path, model_type, config)
            self.model.to(self.device)
            self.is_sequential = True
        else:
            self.model = load_baseline_model(model_path)
            self.is_sequential = False

        # get preprocessing parameters from config
        if config is not None:
            data_config = config.get("data", {})
            self.seq_len = data_config.get("seq_len", 48)
            self.resample_frequency = data_config.get("resample_frequency", "h")
            self.normalize = data_config.get("normalize", True)
            self.normalization_method = data_config.get("normalization_method", "standardize")
            self.normalize_per_series = data_config.get("normalize_per_series", False)
        else:
            self.seq_len = 48
            self.resample_frequency = "h"
            self.normalize = True
            self.normalization_method = "standardize"
            self.normalize_per_series = False

    def predict_fake_probability(
        self,
        time_series: pd.DataFrame,
        id_column: str = "id",
        timestamp_column: str = "timestamp",
    ) -> Dict[str, Union[float, str]]:
        """
        Predict fake probability for a time series.

        Parameters
        ----------
        time_series : pd.DataFrame
            Time series data with columns: id, timestamp, views, likes, comments, shares
        id_column : str
            Name of the ID column
        timestamp_column : str
            Name of the timestamp column

        Returns
        -------
        dict
            Dictionary with 'score' (probability), 'label' (normal/fake), and 'is_fake' (bool)
        """
        # preprocess
        df_preprocessed = preprocess_time_series(
            time_series,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_timezone="UTC",
            resample_frequency=self.resample_frequency,
            handle_missing=True,
        )

        if self.is_sequential:
            # prepare for sequential model
            X, _ = prepare_for_sequential_model(
                df_preprocessed,
                seq_len=self.seq_len,
                id_column=id_column,
                timestamp_column=timestamp_column,
                normalize=self.normalize,
                normalization_method=self.normalization_method,
                normalize_per_series=self.normalize_per_series,
            )

            if len(X) == 0:
                raise ValueError(
                    f"Time series is too short. Need at least {self.seq_len} time steps."
                )

            # use the last sequence (most recent)
            X = X[-1:]

            # predict
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)

                if self.model_type == "autoencoder":
                    # get anomaly score
                    scores = self.model.get_anomaly_scores(X_tensor)
                    score = scores[0].item()
                    # convert score to probability (higher score = more fake)
                    # normalize to [0, 1] range
                    score = min(max(score, 0), 1)  # clip to [0, 1]
                else:
                    # get class probabilities
                    logits = self.model(X_tensor)
                    probs = torch.softmax(logits, dim=1)
                    fake_prob = probs[0, 1].item()  # probability of fake class
                    score = fake_prob

        else:
            # prepare for baseline model
            # get expected number of features from the model's scaler
            expected_n_features = getattr(self.model.scaler, 'n_features_in_', None)
            X = prepare_for_baseline_model(
                df_preprocessed, 
                id_column=id_column,
                expected_n_features=expected_n_features
            )

            if len(X) == 0:
                raise ValueError("No features extracted from time series.")

            # ensure correct number of features
            if expected_n_features is not None and X.shape[1] != expected_n_features:
                # pad or truncate features to match expected size
                if X.shape[1] < expected_n_features:
                    # pad with zeros
                    padding = np.zeros((X.shape[0], expected_n_features - X.shape[1]))
                    X = np.hstack([X, padding])
                else:
                    # truncate (shouldn't happen, but just in case)
                    X = X[:, :expected_n_features]

            # predict
            y_proba = self.model.predict_proba(X)
            if y_proba.ndim > 1:
                score = y_proba[0, 1] if y_proba.shape[1] > 1 else y_proba[0, 0]
            else:
                score = y_proba[0]

        # make decision
        is_fake = score >= self.threshold
        label = "fake" if is_fake else "normal"

        return {
            "score": float(score),
            "label": label,
            "is_fake": bool(is_fake),
        }

    def predict_batch(
        self,
        time_series_list: List[pd.DataFrame],
        id_column: str = "id",
        timestamp_column: str = "timestamp",
    ) -> List[Dict[str, Union[float, str]]]:
        """
        Predict fake probability for multiple time series.

        Parameters
        ----------
        time_series_list : list of pd.DataFrame
            List of time series DataFrames
        id_column : str
            Name of the ID column
        timestamp_column : str
            Name of the timestamp column

        Returns
        -------
        list
            List of prediction dictionaries
        """
        results = []
        for ts in time_series_list:
            try:
                result = self.predict_fake_probability(ts, id_column, timestamp_column)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "score": None, "label": None, "is_fake": None})

        return results


def predict_fake_probability(
    time_series: pd.DataFrame,
    model_path: str,
    model_type: str,
    config: Optional[Dict] = None,
    threshold: float = 0.5,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
) -> Dict[str, Union[float, str]]:
    """
    Simple function to predict fake probability for a time series.

    Parameters
    ----------
    time_series : pd.DataFrame
        Time series data with columns: id, timestamp, views, likes, comments, shares
    model_path : str
        Path to the saved model
    model_type : str
        Type of model: 'baseline' or one of 'lstm', 'tcn', 'autoencoder'
    config : dict, optional
        Configuration dictionary (required for sequential models)
    threshold : float
        Decision threshold for binary classification
    id_column : str
        Name of the ID column
    timestamp_column : str
        Name of the timestamp column

    Returns
    -------
    dict
        Dictionary with 'score' (probability), 'label' (normal/fake), and 'is_fake' (bool)
    """
    pipeline = InferencePipeline(model_path, model_type, config, threshold)
    return pipeline.predict_fake_probability(time_series, id_column, timestamp_column)


def predict_from_series(
    series: List[float],
    model_path: Optional[str] = None,
    model_type: Optional[str] = None,
    config: Optional[Dict] = None,
    threshold: float = 0.5,
    metric_name: str = "views",
) -> Dict[str, Union[float, str]]:
    """
    Predict fake probability from a simple list of numeric values.

    This function is designed for quick testing and manual inference.
    It converts a list of values into a DataFrame and makes a prediction.

    Parameters
    ----------
    series : list of float
        List of numeric values representing a time series (e.g., views over time)
    model_path : str, optional
        Path to the saved model. If None, tries to find a default model.
    model_type : str, optional
        Type of model: 'baseline' or one of 'lstm', 'tcn', 'autoencoder'.
        If None, tries to auto-detect from model_path.
    config : dict, optional
        Configuration dictionary (required for sequential models).
        If None, loads from default config file.
    threshold : float
        Decision threshold for binary classification
    metric_name : str
        Name of the metric column (default: 'views').
        The function will create views, likes, comments, shares based on this.

    Returns
    -------
    dict
        Dictionary with 'score' (probability), 'label' (normal/fake), and 'is_fake' (bool)

    Examples
    --------
    >>> sample = [10, 15, 13, 14, 13, 200, 350, 400, 380, 12, 13]
    >>> result = predict_from_series(sample)
    >>> print(result)
    {'score': 0.85, 'label': 'fake', 'is_fake': True}
    """
    from datetime import datetime, timedelta
    from pathlib import Path
    from src.utils.config import load_config

    # convert series to DataFrame
    n_points = len(series)
    if n_points == 0:
        raise ValueError("Series cannot be empty")

    # create timestamps (hourly intervals)
    start_time = datetime.now() - timedelta(hours=n_points)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]

    # create DataFrame with all engagement metrics
    # for simplicity, we use the series values for views and derive other metrics
    df = pd.DataFrame({
        "id": ["test_series"] * n_points,
        "timestamp": timestamps,
        "views": series,
        "likes": [int(v * 0.1) for v in series],  # approximate 10% like rate
        "comments": [int(v * 0.02) for v in series],  # approximate 2% comment rate
        "shares": [int(v * 0.01) for v in series],  # approximate 1% share rate
    })

    # auto-detect model if not provided
    if model_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        # try to find a default model
        baseline_dir = project_root / "models" / "baselines"
        sequential_dir = project_root / "models" / "sequential"

        # prefer baseline models for simplicity
        if baseline_dir.exists():
            baseline_models = list(baseline_dir.glob("*.pkl"))
            if baseline_models:
                model_path = str(baseline_models[0])
                if model_type is None:
                    # try to infer from filename
                    model_name = baseline_models[0].stem
                    if "random_forest" in model_name:
                        model_type = "random_forest"
                    elif "logistic" in model_name:
                        model_type = "logistic_regression"
                    elif "isolation" in model_name:
                        model_type = "isolation_forest"
                    elif "lof" in model_name:
                        model_type = "lof"
                    else:
                        model_type = "random_forest"  # default
        elif sequential_dir.exists():
            sequential_models = list(sequential_dir.glob("*_best.pth"))
            if sequential_models:
                model_path = str(sequential_models[0])
                if model_type is None:
                    model_name = sequential_models[0].stem
                    if "lstm" in model_name:
                        model_type = "lstm"
                    elif "tcn" in model_name:
                        model_type = "tcn"
                    elif "autoencoder" in model_name:
                        model_type = "autoencoder"
                    else:
                        model_type = "lstm"  # default

        if model_path is None:
            raise FileNotFoundError(
                "No model found. Please train a model first or provide model_path."
            )

    # load config if not provided and needed
    if config is None and model_type in ["lstm", "tcn", "autoencoder"]:
        try:
            config = load_config()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Config file required for sequential models. "
                "Please provide config or ensure config/config.yaml exists."
            )

    # determine if model is baseline or sequential
    # baseline models: logistic_regression, random_forest, isolation_forest, lof
    # sequential models: lstm, tcn, autoencoder
    if model_type in ["lstm", "tcn", "autoencoder"]:
        # sequential model - config is required
        if config is None:
            config = load_config()
    else:
        # baseline model - no config needed
        config = None

    # make prediction
    return predict_fake_probability(
        df,
        model_path=model_path,
        model_type=model_type or "random_forest",
        config=config,
        threshold=threshold,
    )

