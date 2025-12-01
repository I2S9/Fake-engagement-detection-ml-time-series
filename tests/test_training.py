"""
Unit tests for training functions.

Tests minimal training on small datasets to verify training pipeline works.
"""

import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import tempfile
import os

from src.data.simulate_timeseries import generate_dataset
from src.data.preprocess import load_and_preprocess
from src.data.sequence_preparation import prepare_sequences_for_training
from src.data.dataset import create_dataloaders_from_dict
from src.features.temporal_features import extract_temporal_features
from src.training.train import (
    prepare_data,
    train_baseline_model,
    train_sequential_model,
    train_model_from_config,
)
from src.training.evaluate import compute_metrics, evaluate_sequential_model
from src.models.lstm import create_lstm_model
from src.models.tcn import create_tcn_model
from src.models.autoencoder import create_autoencoder_model
from src.utils.config import update_config_with_data


class TestDataPreparation:
    """Tests for data preparation functions."""

    def test_prepare_data(self):
        """Test data preparation for training."""
        # create features dataframe
        df = pd.DataFrame(
            {
                "id": ["v1", "v2", "v3", "v4", "v5"],
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [10, 20, 30, 40, 50],
                "label": ["normal", "normal", "normal", "fake", "fake"],
            }
        )

        X_train, X_test, y_train, y_test, feature_names = prepare_data(df, test_size=0.4, random_state=42)

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert len(feature_names) == 2
        assert (y_train == 0).sum() + (y_train == 1).sum() == len(y_train)


class TestBaselineTraining:
    """Tests for baseline model training."""

    def test_train_baseline_model(self):
        """Test training a baseline model."""
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)

        model = train_baseline_model(
            "logistic_regression",
            X_train,
            y_train,
            model_kwargs={"max_iter": 100},
        )

        assert model.is_fitted
        y_pred = model.predict(X_train[:5])
        assert len(y_pred) == 5

    def test_train_baseline_with_save(self):
        """Test training and saving baseline model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            X_train = np.random.rand(30, 5)
            y_train = np.random.randint(0, 2, 30)

            save_path = os.path.join(tmpdir, "test_model.pkl")
            model = train_baseline_model(
                "random_forest",
                X_train,
                y_train,
                model_kwargs={"n_estimators": 10},
                save_path=save_path,
            )

            assert os.path.exists(save_path)
            assert model.is_fitted


class TestSequentialTraining:
    """Tests for sequential model training."""

    def test_train_lstm_minimal(self):
        """Test minimal LSTM training."""
        # create small dataset
        batch_size = 4
        seq_len = 12
        input_size = 4
        num_samples = 20

        X_train = torch.randn(num_samples, seq_len, input_size)
        y_train = torch.randint(0, 2, (num_samples,))
        X_val = torch.randn(5, seq_len, input_size)
        y_val = torch.randint(0, 2, (5,))

        train_dataset = [(X_train[i], y_train[i]) for i in range(num_samples)]
        val_dataset = [(X_val[i], y_val[i]) for i in range(5)]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = create_lstm_model(
            {
                "input_size": input_size,
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "num_classes": 2,
            }
        )

        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # train for one epoch
        model.train()
        for batch in train_loader:
            X, y = batch
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            break  # just one batch for test

        # verify model works
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(val_loader))
            X_test, y_test = test_batch
            output = model(X_test)
            assert output.shape == (len(X_test), 2)

    def test_train_tcn_minimal(self):
        """Test minimal TCN training."""
        batch_size = 4
        seq_len = 12
        input_size = 4

        X = torch.randn(10, seq_len, input_size)
        y = torch.randint(0, 2, (10,))

        dataset = [(X[i], y[i]) for i in range(10)]
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = create_tcn_model(
            {
                "input_size": input_size,
                "num_channels": [16],
                "kernel_size": 3,
                "dropout": 0.0,
                "num_classes": 2,
            }
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # one training step
        model.train()
        batch = next(iter(loader))
        X_batch, y_batch = batch
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        # verify
        assert not torch.isnan(loss)

    def test_train_autoencoder_minimal(self):
        """Test minimal Autoencoder training."""
        batch_size = 4
        seq_len = 12
        input_size = 4

        X = torch.randn(10, seq_len, input_size)
        y = torch.randint(0, 2, (10,))

        dataset = [(X[i], y[i]) for i in range(10)]
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = create_autoencoder_model(
            {
                "input_size": input_size,
                "seq_len": seq_len,
                "encoder_hidden_sizes": [16],
                "decoder_hidden_sizes": [16],
                "latent_size": 8,
                "dropout": 0.0,
            }
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # one training step
        model.train()
        batch = next(iter(loader))
        X_batch, _ = batch
        optimizer.zero_grad()
        X_recon, _ = model(X_batch)
        loss = criterion(X_recon, X_batch)
        loss.backward()
        optimizer.step()

        # verify
        assert not torch.isnan(loss)
        assert X_recon.shape == X_batch.shape


class TestTrainingPipeline:
    """Tests for complete training pipeline."""

    def test_end_to_end_baseline_training(self):
        """Test end-to-end baseline training pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # generate small dataset
            data_path = os.path.join(tmpdir, "test_data.parquet")
            generate_dataset(
                n_normal=10,
                n_fake=5,
                length_days=3,
                frequency="H",
                output_path=data_path,
                random_seed=42,
            )

            # load and extract features
            df = load_and_preprocess(data_path)
            features_df = extract_temporal_features(df, aggregate_per_id=True)

            # prepare data
            X_train, X_test, y_train, y_test, _ = prepare_data(features_df, test_size=0.3, random_state=42)

            # train model
            model = train_baseline_model(
                "logistic_regression",
                X_train,
                y_train,
                model_kwargs={"max_iter": 50},
            )

            # evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            metrics = compute_metrics(y_test, y_pred, y_proba)

            assert "auc" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics

    def test_end_to_end_sequential_training(self):
        """Test end-to-end sequential training pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # generate small dataset
            data_path = os.path.join(tmpdir, "test_data.parquet")
            generate_dataset(
                n_normal=8,
                n_fake=4,
                length_days=2,
                frequency="H",
                output_path=data_path,
                random_seed=42,
            )

            # load and prepare sequences
            df = load_and_preprocess(data_path)
            sequence_data = prepare_sequences_for_training(
                df,
                seq_len=24,
                normalize=True,
                test_size=0.3,
                val_size=0.2,
                random_state=42,
            )

            # create dataloaders
            dataloaders = create_dataloaders_from_dict(
                sequence_data,
                batch_size=4,
                shuffle_train=False,
            )

            # create model
            input_size = len(sequence_data["feature_names"])
            config = {
                "lstm": {
                    "input_size": input_size,
                    "hidden_size": 16,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "num_classes": 2,
                },
                "data": {"seq_len": 24},
            }

            model = create_lstm_model(config["lstm"])
            device = torch.device("cpu")

            # minimal training (just verify it works)
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # one batch
            batch = next(iter(dataloaders["train"]))
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # verify
            assert not torch.isnan(loss)
            assert logits.shape[0] == X.shape[0]

    def test_training_metrics_computation(self):
        """Test that training produces valid metrics."""
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        X_test = np.random.rand(20, 5)
        y_test = np.random.randint(0, 2, 20)

        model = train_baseline_model("logistic_regression", X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = compute_metrics(y_test, y_pred, y_proba)

        assert isinstance(metrics, dict)
        assert "auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

