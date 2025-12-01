"""
Unit tests for data loading and preprocessing functions.
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
import tempfile
import os

from src.data.simulate_timeseries import (
    generate_normal_timeseries,
    generate_fake_timeseries,
    generate_dataset,
)
from src.data.load_data import load_data, load_parquet, load_csv
from src.data.preprocess import (
    handle_timezone,
    sort_by_id_and_timestamp,
    resample_timeseries,
    handle_missing_values,
    normalize_metrics,
    load_and_preprocess,
)


class TestDataGeneration:
    """Tests for data generation functions."""

    def test_generate_normal_timeseries(self):
        """Test normal time series generation."""
        start_date = datetime.now() - timedelta(days=7)
        df = generate_normal_timeseries(
            start_date=start_date,
            length_days=7,
            frequency="H",
            video_id="test_001",
            random_seed=42,
        )

        assert len(df) > 0
        assert "id" in df.columns
        assert "timestamp" in df.columns
        assert "views" in df.columns
        assert "likes" in df.columns
        assert "comments" in df.columns
        assert "shares" in df.columns
        assert "label" in df.columns
        assert df["label"].iloc[0] == "normal"
        assert (df["views"] >= 0).all()
        assert (df["likes"] >= 0).all()

    def test_generate_fake_timeseries(self):
        """Test fake time series generation."""
        start_date = datetime.now() - timedelta(days=7)
        df = generate_fake_timeseries(
            start_date=start_date,
            length_days=7,
            frequency="H",
            video_id="test_002",
            fake_pattern="burst",
            random_seed=42,
        )

        assert len(df) > 0
        assert df["label"].iloc[0] == "fake"
        assert (df["views"] >= 0).all()

    def test_generate_dataset(self):
        """Test dataset generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_dataset.parquet")
            df = generate_dataset(
                n_normal=5,
                n_fake=2,
                length_days=3,
                frequency="H",
                output_path=output_path,
                random_seed=42,
            )

            assert len(df) > 0
            assert df["id"].nunique() == 7
            assert "normal" in df["label"].values
            assert "fake" in df["label"].values
            assert os.path.exists(output_path)


class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_parquet(self):
        """Test loading parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test data
            df_test = pd.DataFrame(
                {
                    "id": ["v1", "v2"],
                    "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
                    "views": [100, 200],
                    "likes": [10, 20],
                    "comments": [5, 10],
                    "shares": [2, 4],
                    "label": ["normal", "fake"],
                }
            )

            file_path = os.path.join(tmpdir, "test.parquet")
            df_test.to_parquet(file_path, index=False)

            # load
            df_loaded = load_parquet(file_path)

            assert len(df_loaded) == len(df_test)
            assert list(df_loaded.columns) == list(df_test.columns)

    def test_load_csv(self):
        """Test loading CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # create test data
            df_test = pd.DataFrame(
                {
                    "id": ["v1", "v2"],
                    "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
                    "views": [100, 200],
                    "likes": [10, 20],
                    "comments": [5, 10],
                    "shares": [2, 4],
                    "label": ["normal", "fake"],
                }
            )

            file_path = os.path.join(tmpdir, "test.csv")
            df_test.to_csv(file_path, index=False)

            # load
            df_loaded = load_csv(file_path, date_column="timestamp")

            assert len(df_loaded) == len(df_test)
            assert pd.api.types.is_datetime64_any_dtype(df_loaded["timestamp"])

    def test_load_data_auto_detect(self):
        """Test load_data with auto-detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df_test = pd.DataFrame(
                {
                    "id": ["v1"],
                    "timestamp": pd.date_range("2024-01-01", periods=1, freq="h"),
                    "views": [100],
                    "likes": [10],
                    "comments": [5],
                    "shares": [2],
                    "label": ["normal"],
                }
            )

            # test parquet
            parquet_path = os.path.join(tmpdir, "test.parquet")
            df_test.to_parquet(parquet_path, index=False)
            df_loaded = load_data(parquet_path)
            assert len(df_loaded) == 1

            # test csv
            csv_path = os.path.join(tmpdir, "test.csv")
            df_test.to_csv(csv_path, index=False)
            df_loaded = load_data(csv_path)
            assert len(df_loaded) == 1


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_handle_timezone(self):
        """Test timezone handling."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
                "views": [100] * 5,
            }
        )

        df_result = handle_timezone(df, target_timezone="UTC")
        assert pd.api.types.is_datetime64_any_dtype(df_result["timestamp"])
        if df_result["timestamp"].dt.tz is not None:
            assert str(df_result["timestamp"].dt.tz) == "UTC"

    def test_sort_by_id_and_timestamp(self):
        """Test sorting by ID and timestamp."""
        df = pd.DataFrame(
            {
                "id": ["v2", "v1", "v2", "v1"],
                "timestamp": [
                    datetime(2024, 1, 1, 2),
                    datetime(2024, 1, 1, 1),
                    datetime(2024, 1, 1, 1),
                    datetime(2024, 1, 1, 2),
                ],
                "views": [100, 200, 300, 400],
            }
        )

        df_sorted = sort_by_id_and_timestamp(df)

        assert df_sorted["id"].iloc[0] == "v1"
        assert df_sorted["id"].iloc[1] == "v1"
        assert df_sorted["timestamp"].iloc[0] < df_sorted["timestamp"].iloc[1]

    def test_resample_timeseries(self):
        """Test time series resampling."""
        df = pd.DataFrame(
            {
                "id": ["v1"] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="30min"),
                "views": range(10),
                "likes": range(10),
            }
        )

        df_resampled = resample_timeseries(
            df, frequency="h", aggregation_method="sum"
        )

        assert len(df_resampled) <= len(df)
        assert "views" in df_resampled.columns

    def test_handle_missing_values(self):
        """Test missing value handling."""
        df = pd.DataFrame(
            {
                "views": [100, np.nan, 300, 400],
                "likes": [10, 20, np.nan, 40],
                "comments": [5, 10, 15, 20],
            }
        )

        df_filled = handle_missing_values(df, method="forward")

        assert df_filled["views"].isna().sum() == 0
        assert df_filled["likes"].isna().sum() == 0

    def test_normalize_metrics(self):
        """Test metric normalization."""
        df = pd.DataFrame(
            {
                "id": ["v1", "v1", "v2", "v2"],
                "views": [100, 200, 300, 400],
                "likes": [10, 20, 30, 40],
            }
        )

        df_normalized = normalize_metrics(
            df, metric_columns=["views"], method="standardize", group_by_id=True
        )

        assert "views_normalized" in df_normalized.columns
        assert not df_normalized["views_normalized"].isna().all()

    def test_load_and_preprocess(self):
        """Test complete preprocessing pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # generate test data
            output_path = os.path.join(tmpdir, "test_data.parquet")
            generate_dataset(
                n_normal=3,
                n_fake=1,
                length_days=2,
                frequency="H",
                output_path=output_path,
                random_seed=42,
            )

            # load and preprocess
            df = load_and_preprocess(
                file_path=output_path,
                target_timezone="UTC",
                resample_frequency="h",
                handle_missing=True,
                normalize=False,
            )

            assert len(df) > 0
            assert "id" in df.columns
            assert "timestamp" in df.columns
            assert "views" in df.columns
            assert df["views"].isna().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

