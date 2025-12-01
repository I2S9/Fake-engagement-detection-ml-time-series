"""
Unit tests for model implementations.

Tests forward pass, output shapes, and basic model functionality.
"""

import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.lstm import LSTMModel, create_lstm_model
from src.models.tcn import TCNModel, create_tcn_model
from src.models.autoencoder import AutoencoderModel, create_autoencoder_model
from src.models.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    IsolationForestBaseline,
    LOFBaseline,
    create_baseline_model,
)


class TestLSTMModel:
    """Tests for LSTM model."""

    def test_lstm_forward_pass(self):
        """Test LSTM forward pass."""
        batch_size = 4
        seq_len = 24
        input_size = 4
        num_classes = 2

        model = LSTMModel(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            bidirectional=False,
            num_classes=num_classes,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any()

    def test_lstm_bidirectional(self):
        """Test bidirectional LSTM."""
        model = LSTMModel(
            input_size=4,
            hidden_size=32,
            num_layers=1,
            bidirectional=True,
            num_classes=2,
        )

        x = torch.randn(2, 24, 4)
        output = model(x)

        assert output.shape == (2, 2)

    def test_lstm_predict_proba(self):
        """Test LSTM probability prediction."""
        model = LSTMModel(input_size=4, hidden_size=32, num_classes=2)
        x = torch.randn(2, 24, 4)

        probs = model.predict_proba(x)

        assert probs.shape == (2, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_create_lstm_model(self):
        """Test LSTM model creation from config."""
        config = {
            "input_size": 4,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "num_classes": 2,
        }

        model = create_lstm_model(config)

        assert isinstance(model, LSTMModel)
        assert model.input_size == 4
        assert model.hidden_size == 64


class TestTCNModel:
    """Tests for TCN model."""

    def test_tcn_forward_pass(self):
        """Test TCN forward pass."""
        batch_size = 4
        seq_len = 24
        input_size = 4
        num_classes = 2

        model = TCNModel(
            input_size=input_size,
            num_channels=[32, 32],
            kernel_size=3,
            dropout=0.1,
            num_classes=num_classes,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any()

    def test_tcn_predict_proba(self):
        """Test TCN probability prediction."""
        model = TCNModel(input_size=4, num_channels=[32], num_classes=2)
        x = torch.randn(2, 24, 4)

        probs = model.predict_proba(x)

        assert probs.shape == (2, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_create_tcn_model(self):
        """Test TCN model creation from config."""
        config = {
            "input_size": 4,
            "num_channels": [64, 64],
            "kernel_size": 3,
            "dropout": 0.2,
            "num_classes": 2,
        }

        model = create_tcn_model(config)

        assert isinstance(model, TCNModel)
        assert model.input_size == 4


class TestAutoencoderModel:
    """Tests for Autoencoder model."""

    def test_autoencoder_forward_pass(self):
        """Test Autoencoder forward pass."""
        batch_size = 4
        seq_len = 24
        input_size = 4

        model = AutoencoderModel(
            input_size=input_size,
            seq_len=seq_len,
            encoder_hidden_sizes=[32, 16],
            decoder_hidden_sizes=[16, 32],
            latent_size=8,
            dropout=0.1,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        x_recon, z = model(x)

        assert x_recon.shape == x.shape
        assert z.shape == (batch_size, 8)
        assert not torch.isnan(x_recon).any()
        assert not torch.isnan(z).any()

    def test_autoencoder_encode_decode(self):
        """Test encoder and decoder separately."""
        model = AutoencoderModel(
            input_size=4,
            seq_len=24,
            encoder_hidden_sizes=[32, 16],
            decoder_hidden_sizes=[16, 32],
            latent_size=8,
        )

        x = torch.randn(2, 24, 4)
        z = model.encode(x)
        x_recon = model.decode(z)

        assert z.shape == (2, 8)
        assert x_recon.shape == x.shape

    def test_autoencoder_reconstruction_error(self):
        """Test reconstruction error computation."""
        model = AutoencoderModel(
            input_size=4,
            seq_len=24,
            encoder_hidden_sizes=[32],
            decoder_hidden_sizes=[32],
            latent_size=8,
        )

        x = torch.randn(3, 24, 4)
        x_recon, _ = model(x)

        errors = model.compute_reconstruction_error(x, x_recon)

        assert errors.shape == (3,)
        assert (errors >= 0).all()

    def test_autoencoder_anomaly_scores(self):
        """Test anomaly score computation."""
        model = AutoencoderModel(
            input_size=4,
            seq_len=24,
            encoder_hidden_sizes=[32],
            decoder_hidden_sizes=[32],
            latent_size=8,
        )

        x = torch.randn(2, 24, 4)
        scores = model.get_anomaly_scores(x)

        assert scores.shape == (2,)
        assert not torch.isnan(scores).any()

    def test_create_autoencoder_model(self):
        """Test Autoencoder model creation from config."""
        config = {
            "input_size": 4,
            "seq_len": 24,
            "encoder_hidden_sizes": [32, 16],
            "decoder_hidden_sizes": [16, 32],
            "latent_size": 8,
            "dropout": 0.2,
        }

        model = create_autoencoder_model(config)

        assert isinstance(model, AutoencoderModel)
        assert model.input_size == 4
        assert model.seq_len == 24


class TestBaselineModels:
    """Tests for baseline models."""

    def test_logistic_regression_baseline(self):
        """Test Logistic Regression baseline."""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegressionBaseline()
        model.fit(X, y)

        y_pred = model.predict(X[:10])
        y_proba = model.predict_proba(X[:10])

        assert len(y_pred) == 10
        assert y_proba.shape == (10, 2)
        assert (y_pred >= 0).all() and (y_pred <= 1).all()

    def test_random_forest_baseline(self):
        """Test Random Forest baseline."""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        model = RandomForestBaseline(n_estimators=10)
        model.fit(X, y)

        y_pred = model.predict(X[:10])
        y_proba = model.predict_proba(X[:10])

        assert len(y_pred) == 10
        assert y_proba.shape == (10, 2)

        # test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 10

    def test_isolation_forest_baseline(self):
        """Test Isolation Forest baseline."""
        X = np.random.rand(100, 10)

        model = IsolationForestBaseline(contamination=0.1)
        model.fit(X)

        y_pred = model.predict(X[:10])
        y_proba = model.predict_proba(X[:10])
        scores = model.get_anomaly_scores(X[:10])

        assert len(y_pred) == 10
        assert y_proba.shape == (10, 2)
        assert len(scores) == 10

    def test_lof_baseline(self):
        """Test LOF baseline."""
        X = np.random.rand(100, 10)

        model = LOFBaseline(n_neighbors=10, contamination=0.1)
        model.fit(X)

        y_pred = model.predict(X[:10])
        y_proba = model.predict_proba(X[:10])
        scores = model.get_anomaly_scores(X[:10])

        assert len(y_pred) == 10
        assert y_proba.shape == (10, 2)
        assert len(scores) == 10

    def test_create_baseline_model(self):
        """Test baseline model factory function."""
        model = create_baseline_model("logistic_regression")
        assert isinstance(model, LogisticRegressionBaseline)

        model = create_baseline_model("random_forest")
        assert isinstance(model, RandomForestBaseline)

        model = create_baseline_model("isolation_forest")
        assert isinstance(model, IsolationForestBaseline)

        model = create_baseline_model("lof")
        assert isinstance(model, LOFBaseline)


class TestModelShapes:
    """Tests for model output shapes."""

    def test_all_models_output_shapes(self):
        """Test that all models produce correct output shapes."""
        batch_size = 8
        seq_len = 48
        input_size = 4
        num_classes = 2

        # LSTM
        lstm = LSTMModel(input_size=input_size, hidden_size=32, num_classes=num_classes)
        x_lstm = torch.randn(batch_size, seq_len, input_size)
        out_lstm = lstm(x_lstm)
        assert out_lstm.shape == (batch_size, num_classes)

        # TCN
        tcn = TCNModel(input_size=input_size, num_channels=[32], num_classes=num_classes)
        x_tcn = torch.randn(batch_size, seq_len, input_size)
        out_tcn = tcn(x_tcn)
        assert out_tcn.shape == (batch_size, num_classes)

        # Autoencoder
        ae = AutoencoderModel(
            input_size=input_size, seq_len=seq_len, encoder_hidden_sizes=[32], decoder_hidden_sizes=[32], latent_size=8
        )
        x_ae = torch.randn(batch_size, seq_len, input_size)
        recon_ae, z_ae = ae(x_ae)
        assert recon_ae.shape == x_ae.shape
        assert z_ae.shape == (batch_size, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

