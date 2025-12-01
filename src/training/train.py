"""
Training pipeline for baseline and sequential models.

This module provides functions to train baseline models and sequential deep learning
models (LSTM, TCN, Autoencoder) with early stopping and checkpoint saving.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import json
from datetime import datetime

from src.models.baselines import create_baseline_model, BaselineModel
from src.models.lstm import create_lstm_model
from src.models.tcn import create_tcn_model
from src.models.autoencoder import create_autoencoder_model


def prepare_data(
    features_df: pd.DataFrame,
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Prepare data for training by splitting features and labels.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features and labels
    label_column : str
        Name of the label column
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, feature_names
    """
    # separate features and labels
    feature_columns = [col for col in features_df.columns if col not in [label_column, "id"]]
    X = features_df[feature_columns].values
    y = features_df[label_column].values

    # convert labels to binary (normal=0, fake=1)
    y_binary = np.where(y == "fake", 1, 0)

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
    )

    return X_train, X_test, y_train, y_test, feature_columns


def train_baseline_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_kwargs: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> BaselineModel:
    """
    Train a baseline model.

    Parameters
    ----------
    model_type : str
        Type of model to train
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    model_kwargs : dict, optional
        Additional arguments for the model
    save_path : str, optional
        Path to save the trained model

    Returns
    -------
    BaselineModel
        Trained model
    """
    if model_kwargs is None:
        model_kwargs = {}

    # create and train model
    model = create_baseline_model(model_type, **model_kwargs)

    print(f"Training {model_type}...")
    model.fit(X_train, y_train)
    print(f"Training completed.")

    # save model if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Model saved to {save_path}")

    return model


def train_multiple_baselines(
    features_df: pd.DataFrame,
    model_types: list,
    test_size: float = 0.2,
    random_state: int = 42,
    model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, Tuple[BaselineModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Train multiple baseline models.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features and labels
    model_types : list
        List of model types to train
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    model_kwargs : dict, optional
        Dictionary mapping model_type to kwargs for each model
    save_dir : str, optional
        Directory to save trained models

    Returns
    -------
    dict
        Dictionary mapping model_type to (model, X_test, y_test, y_pred, y_proba)
    """
    # prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        features_df, test_size=test_size, random_state=random_state
    )

    if model_kwargs is None:
        model_kwargs = {}

    results = {}

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type}")
        print(f"{'='*60}")

        try:
            # get model-specific kwargs
            kwargs = model_kwargs.get(model_type, {})

            # train model
            save_path = None
            if save_dir is not None:
                save_path = os.path.join(save_dir, f"{model_type}.pkl")

            model = train_baseline_model(
                model_type, X_train, y_train, model_kwargs=kwargs, save_path=save_path
            )

            # make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            results[model_type] = (model, X_test, y_test, y_pred, y_proba)

            print(f"{model_type} training completed successfully.")

        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            continue

    return results


def train_sequential_model(
    model_type: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    save_dir: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a sequential model (LSTM, TCN, or Autoencoder) with early stopping.

    Parameters
    ----------
    model_type : str
        Type of model: 'lstm', 'tcn', or 'autoencoder'
    model : nn.Module
        PyTorch model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    config : dict
        Configuration dictionary with training parameters
    device : torch.device
        Device to train on
    save_dir : str, optional
        Directory to save checkpoints

    Returns
    -------
    tuple
        (best_model, history) where history contains training metrics
    """
    model = model.to(device)

    # get training config
    model_config = config.get(model_type, {})
    learning_rate = model_config.get("learning_rate", 0.001)
    weight_decay = model_config.get("weight_decay", 0.0001)
    num_epochs = model_config.get("num_epochs", 50)
    early_stopping_patience = model_config.get("early_stopping_patience", 10)

    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # setup loss function
    if model_type == "autoencoder":
        criterion = nn.MSELoss()
        metric_key = "loss"
    else:
        criterion = nn.CrossEntropyLoss()
        metric_key = "loss"

    # training history
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    # early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        # training phase
        model.train()
        train_metrics = {"loss": [], "accuracy": []}

        for batch in train_loader:
            if model_type == "autoencoder":
                result = model.train_step(batch, criterion, optimizer, device)
            else:
                result = model.train_step(batch, criterion, optimizer, device)

            train_metrics["loss"].append(result["loss"])
            if "accuracy" in result:
                train_metrics["accuracy"].append(result["accuracy"])

        avg_train_loss = np.mean(train_metrics["loss"])
        avg_train_acc = np.mean(train_metrics["accuracy"]) if train_metrics["accuracy"] else 0.0

        # validation phase
        model.eval()
        val_metrics = {"loss": [], "accuracy": []}

        with torch.no_grad():
            for batch in val_loader:
                if model_type == "autoencoder":
                    result = model.eval_step(batch, criterion, device)
                else:
                    result = model.eval_step(batch, criterion, device)

                val_metrics["loss"].append(result["loss"])
                if "accuracy" in result:
                    val_metrics["accuracy"].append(result["accuracy"])

        avg_val_loss = np.mean(val_metrics["loss"])
        avg_val_acc = np.mean(val_metrics["accuracy"]) if val_metrics["accuracy"] else 0.0

        # update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_accuracy"].append(avg_train_acc)
        history["val_accuracy"].append(avg_val_acc)

        # print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )
            if avg_train_acc > 0:
                print(f"  Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}")

        # early stopping and checkpoint saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()

            # save checkpoint
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"{model_type}_best.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "history": history,
                    },
                    checkpoint_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    return model, history


def train_model_from_config(
    model_type: str,
    dataloaders: Dict[str, DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    save_dir: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a model from configuration.

    Parameters
    ----------
    model_type : str
        Type of model: 'lstm', 'tcn', or 'autoencoder'
    dataloaders : dict
        Dictionary with 'train' and 'val' DataLoaders
    config : dict
        Configuration dictionary
    device : torch.device
        Device to train on
    save_dir : str, optional
        Directory to save checkpoints

    Returns
    -------
    tuple
        (trained_model, training_history)
    """
    # get model config and create model
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

    # train model
    model, history = train_sequential_model(
        model_type=model_type,
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=config,
        device=device,
        save_dir=save_dir,
    )

    return model, history

