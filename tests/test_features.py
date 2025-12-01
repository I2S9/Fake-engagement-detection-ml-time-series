"""
Unit tests for feature engineering functions.
"""

import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.temporal_features import (
    compute_rolling_statistics,
    compute_ratios,
    detect_bursts,
    compute_autocorrelation,
    compute_entropy,
    compute_trend_features,
    extract_temporal_features,
)
from src.data.simulate_timeseries import generate_normal_timeseries


class TestRollingStatistics:
    """Tests for rolling statistics computation."""

    def test_compute_rolling_statistics(self):
        """Test rolling statistics computation."""
        series = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        result = compute_rolling_statistics(series, window_sizes=[3, 5])

        assert "rolling_mean_3" in result.columns
        assert "rolling_mean_5" in result.columns
        assert "rolling_std_3" in result.columns
        assert len(result) == len(series)

    def test_rolling_statistics_values(self):
        """Test rolling statistics values are correct."""
        series = pd.Series([10, 20, 30, 40, 50])
        result = compute_rolling_statistics(series, window_sizes=[3])

        assert not result["rolling_mean_3"].isna().all()
        assert result["rolling_mean_3"].iloc[-1] == 40.0  # mean of [30, 40, 50]


class TestRatios:
    """Tests for ratio computation."""

    def test_compute_ratios(self):
        """Test ratio computation."""
        df = pd.DataFrame(
            {
                "views": [100, 200, 300],
                "likes": [10, 20, 30],
                "comments": [5, 10, 15],
            }
        )

        ratios = compute_ratios(df, numerator_cols=["likes", "comments"], denominator_col="views")

        assert "ratio_likes_views" in ratios.columns
        assert "ratio_comments_views" in ratios.columns
        assert np.isclose(ratios["ratio_likes_views"].iloc[0], 0.1, atol=1e-6)

    def test_ratios_division_by_zero(self):
        """Test ratio computation handles zero views."""
        df = pd.DataFrame(
            {
                "views": [0, 100, 200],
                "likes": [10, 20, 30],
            }
        )

        ratios = compute_ratios(df, numerator_cols=["likes"], denominator_col="views")

        # should not raise error, should handle zero division
        assert "ratio_likes_views" in ratios.columns


class TestBurstDetection:
    """Tests for burst detection."""

    def test_detect_bursts(self):
        """Test burst detection."""
        # create series with a peak
        series = pd.Series([10, 15, 20, 100, 25, 20, 15])  # peak at index 3

        bursts = detect_bursts(series, threshold_multiplier=2.0)

        assert "n_peaks" in bursts
        assert "max_value" in bursts
        assert "max_mean_ratio" in bursts
        assert bursts["max_value"] == 100.0

    def test_detect_bursts_no_peaks(self):
        """Test burst detection with no peaks."""
        series = pd.Series([10, 11, 12, 13, 14, 15])

        bursts = detect_bursts(series, threshold_multiplier=2.0)

        assert bursts["n_peaks"] >= 0
        assert bursts["max_value"] == 15.0


class TestAutocorrelation:
    """Tests for autocorrelation computation."""

    def test_compute_autocorrelation(self):
        """Test autocorrelation computation."""
        # create series with some correlation
        series = pd.Series([10, 12, 14, 16, 18, 20, 22, 24])

        autocorr = compute_autocorrelation(series, lags=[1, 2])

        assert "autocorr_lag_1" in autocorr
        assert "autocorr_lag_2" in autocorr
        assert -1 <= autocorr["autocorr_lag_1"] <= 1

    def test_autocorrelation_short_series(self):
        """Test autocorrelation with short series."""
        series = pd.Series([10, 20])

        autocorr = compute_autocorrelation(series, lags=[1, 2])

        assert "autocorr_lag_1" in autocorr
        assert "autocorr_lag_2" in autocorr


class TestEntropy:
    """Tests for entropy computation."""

    def test_compute_entropy(self):
        """Test entropy computation."""
        series = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        entropy_features = compute_entropy(series)

        assert "entropy" in entropy_features
        assert "regularity" in entropy_features
        assert "coefficient_of_variation" in entropy_features
        assert entropy_features["entropy"] >= 0
        assert 0 <= entropy_features["regularity"] <= 1

    def test_entropy_constant_series(self):
        """Test entropy with constant series."""
        series = pd.Series([10, 10, 10, 10, 10])

        entropy_features = compute_entropy(series)

        assert "entropy" in entropy_features
        assert entropy_features["coefficient_of_variation"] == 0.0


class TestTrendFeatures:
    """Tests for trend feature computation."""

    def test_compute_trend_features(self):
        """Test trend feature computation."""
        # increasing trend
        series = pd.Series([10, 20, 30, 40, 50, 60, 70, 80])

        trend = compute_trend_features(series)

        assert "trend_slope" in trend
        assert "trend_strength" in trend
        assert "first_last_ratio" in trend
        assert trend["trend_slope"] > 0  # positive slope
        assert trend["first_last_ratio"] > 1  # last > first

    def test_trend_decreasing(self):
        """Test trend with decreasing series."""
        series = pd.Series([100, 90, 80, 70, 60, 50])

        trend = compute_trend_features(series)

        assert trend["trend_slope"] < 0  # negative slope
        assert trend["first_last_ratio"] < 1  # last < first


class TestFeatureExtraction:
    """Tests for complete feature extraction."""

    def test_extract_temporal_features(self):
        """Test complete feature extraction."""
        # create test data
        start_date = datetime.now() - timedelta(days=2)
        df = generate_normal_timeseries(
            start_date=start_date,
            length_days=2,
            frequency="H",
            video_id="test_001",
            random_seed=42,
        )

        features_df = extract_temporal_features(
            df,
            id_column="id",
            timestamp_column="timestamp",
            window_sizes=[6, 12],
            autocorr_lags=[1, 6],
            aggregate_per_id=True,
        )

        assert len(features_df) > 0
        assert "id" in features_df.columns
        assert len(features_df.columns) > 5  # should have many features

    def test_feature_extraction_dimensions(self):
        """Test feature extraction produces correct dimensions."""
        start_date = datetime.now() - timedelta(days=2)
        df = generate_normal_timeseries(
            start_date=start_date,
            length_days=2,
            frequency="H",
            video_id="test_002",
            random_seed=42,
        )

        features_df = extract_temporal_features(
            df,
            aggregate_per_id=True,
        )

        # should have one row per video ID
        assert len(features_df) == 1
        assert features_df["id"].iloc[0] == "test_002"

    def test_feature_types(self):
        """Test that features have correct types."""
        start_date = datetime.now() - timedelta(days=2)
        df = generate_normal_timeseries(
            start_date=start_date,
            length_days=2,
            frequency="H",
            video_id="test_003",
            random_seed=42,
        )

        features_df = extract_temporal_features(
            df,
            aggregate_per_id=True,
        )

        # check that feature columns are numeric
        feature_cols = [col for col in features_df.columns if col not in ["id", "label"]]
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(features_df[col])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

