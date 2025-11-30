"""
Preprocess time series data for modeling.

This module handles timezone conversion, sorting, resampling, missing values,
and normalization of engagement time series data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List
from pathlib import Path


def handle_timezone(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    target_timezone: Optional[str] = None,
    source_timezone: Optional[str] = None,
) -> pd.DataFrame:
    """
    Handle timezone conversion for timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with timestamp column
    timestamp_column : str
        Name of the timestamp column
    target_timezone : str, optional
        Target timezone (e.g., 'UTC', 'Europe/Paris'). If None, converts to UTC
    source_timezone : str, optional
        Source timezone. If None and timestamp is timezone-naive, assumes UTC

    Returns
    -------
    pd.DataFrame
        DataFrame with timezone-aware timestamps
    """
    df = df.copy()

    if timestamp_column not in df.columns:
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

    if target_timezone is None:
        target_timezone = "UTC"

    # ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # handle timezone
    if df[timestamp_column].dt.tz is None:
        # timezone-naive: localize if source_timezone provided
        if source_timezone is not None:
            df[timestamp_column] = df[timestamp_column].dt.tz_localize(source_timezone)
        else:
            # assume UTC if not specified
            df[timestamp_column] = df[timestamp_column].dt.tz_localize("UTC")
    else:
        # already timezone-aware: convert to target
        pass

    # convert to target timezone
    df[timestamp_column] = df[timestamp_column].dt.tz_convert(target_timezone)

    return df


def sort_by_id_and_timestamp(
    df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Sort DataFrame by id and timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    id_column : str
        Name of the ID column
    timestamp_column : str
        Name of the timestamp column

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame
    """
    df = df.copy()

    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame")
    if timestamp_column not in df.columns:
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

    df = df.sort_values(by=[id_column, timestamp_column]).reset_index(drop=True)
    return df


def resample_timeseries(
    df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    frequency: str = "h",
    aggregation_method: str = "sum",
    fill_method: str = "forward",
) -> pd.DataFrame:
    """
    Resample time series to uniform frequency per ID.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data
    id_column : str
        Name of the ID column
    timestamp_column : str
        Name of the timestamp column
    frequency : str
        Target frequency (e.g., 'h' for hourly, 'D' for daily)
    aggregation_method : str
        Method for aggregating values: 'sum', 'mean', 'max', 'min', 'last'
    fill_method : str
        Method for filling missing values after resampling: 'forward', 'backward', 'zero', 'interpolate'

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with uniform frequency
    """
    df = df.copy()

    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame")
    if timestamp_column not in df.columns:
        raise ValueError(f"Column '{timestamp_column}' not found in DataFrame")

    # set timestamp as index for resampling
    df = df.set_index(timestamp_column)

    # columns to aggregate (exclude id and label)
    metric_columns = ["views", "likes", "comments", "shares"]
    metric_columns = [col for col in metric_columns if col in df.columns]

    # group by id and resample
    resampled_dfs = []
    for video_id, group in df.groupby(id_column):
        # resample
        if aggregation_method == "sum":
            resampled = group[metric_columns].resample(frequency).sum()
        elif aggregation_method == "mean":
            resampled = group[metric_columns].resample(frequency).mean()
        elif aggregation_method == "max":
            resampled = group[metric_columns].resample(frequency).max()
        elif aggregation_method == "min":
            resampled = group[metric_columns].resample(frequency).min()
        elif aggregation_method == "last":
            resampled = group[metric_columns].resample(frequency).last()
        else:
            raise ValueError(
                f"Unknown aggregation_method: {aggregation_method}. "
                "Must be one of: sum, mean, max, min, last"
            )

        # handle non-metric columns (id, label)
        if "label" in group.columns:
            label_value = group["label"].iloc[0]
            resampled["label"] = label_value

        resampled[id_column] = video_id
        resampled_dfs.append(resampled)

    # combine all resampled dataframes
    result_df = pd.concat(resampled_dfs)

    # fill missing values
    if fill_method == "forward":
        result_df[metric_columns] = result_df[metric_columns].ffill()
    elif fill_method == "backward":
        result_df[metric_columns] = result_df[metric_columns].bfill()
    elif fill_method == "zero":
        result_df[metric_columns] = result_df[metric_columns].fillna(0)
    elif fill_method == "interpolate":
        result_df[metric_columns] = result_df[metric_columns].interpolate(method="time")
    else:
        raise ValueError(
            f"Unknown fill_method: {fill_method}. "
            "Must be one of: forward, backward, zero, interpolate"
        )

    # reset index to get timestamp as column
    result_df = result_df.reset_index()

    # ensure non-negative values
    for col in metric_columns:
        result_df[col] = result_df[col].clip(lower=0)

    return result_df


def handle_missing_values(
    df: pd.DataFrame,
    metric_columns: Optional[List[str]] = None,
    method: str = "forward",
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Handle missing values in time series data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    metric_columns : list, optional
        List of metric columns to process. If None, uses default: views, likes, comments, shares
    method : str
        Imputation method: 'forward', 'backward', 'zero', 'interpolate', 'drop'
    drop_threshold : float
        If more than this proportion of values are missing for a column, drop the column

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
    """
    df = df.copy()

    if metric_columns is None:
        metric_columns = ["views", "likes", "comments", "shares"]
        metric_columns = [col for col in metric_columns if col in df.columns]

    # check for columns with too many missing values
    for col in metric_columns:
        if col in df.columns:
            missing_ratio = df[col].isna().sum() / len(df)
            if missing_ratio > drop_threshold:
                print(f"Warning: Dropping column '{col}' with {missing_ratio:.1%} missing values")
                df = df.drop(columns=[col])
                metric_columns.remove(col)

    # handle remaining missing values
    for col in metric_columns:
        if col not in df.columns:
            continue

        if method == "forward":
            df[col] = df[col].ffill()
        elif method == "backward":
            df[col] = df[col].bfill()
        elif method == "zero":
            df[col] = df[col].fillna(0)
        elif method == "interpolate":
            df[col] = df[col].interpolate(method="linear")
        elif method == "drop":
            df = df.dropna(subset=[col])
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Must be one of: forward, backward, zero, interpolate, drop"
            )

    # final check: if still missing, fill with 0
    for col in metric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def normalize_metrics(
    df: pd.DataFrame,
    metric_columns: Optional[List[str]] = None,
    method: str = "standardize",
    group_by_id: bool = True,
    id_column: str = "id",
) -> pd.DataFrame:
    """
    Normalize or standardize engagement metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    metric_columns : list, optional
        List of metric columns to normalize. If None, uses default: views, likes, comments, shares
    method : str
        Normalization method: 'standardize' (z-score), 'minmax' (0-1), 'robust' (median/IQR)
    group_by_id : bool
        If True, normalize separately for each ID (video). If False, normalize globally
    id_column : str
        Name of the ID column (used if group_by_id=True)

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized metrics (original columns preserved with '_normalized' suffix)
    """
    df = df.copy()

    if metric_columns is None:
        metric_columns = ["views", "likes", "comments", "shares"]
        metric_columns = [col for col in metric_columns if col in df.columns]

    if group_by_id and id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found for grouped normalization")

    normalized_columns = []

    if group_by_id:
        # normalize per ID
        for video_id, group in df.groupby(id_column):
            for col in metric_columns:
                if col not in group.columns:
                    continue

                values = group[col].values.astype(float)

                if method == "standardize":
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    if std_val > 0:
                        normalized = (values - mean_val) / std_val
                    else:
                        normalized = values * 0
                elif method == "minmax":
                    min_val = np.min(values)
                    max_val = np.max(values)
                    if max_val > min_val:
                        normalized = (values - min_val) / (max_val - min_val)
                    else:
                        normalized = values * 0
                elif method == "robust":
                    median_val = np.median(values)
                    q75 = np.percentile(values, 75)
                    q25 = np.percentile(values, 25)
                    iqr = q75 - q25
                    if iqr > 0:
                        normalized = (values - median_val) / iqr
                    else:
                        normalized = values * 0
                else:
                    raise ValueError(
                        f"Unknown method: {method}. "
                        "Must be one of: standardize, minmax, robust"
                    )

                col_name = f"{col}_normalized"
                df.loc[group.index, col_name] = normalized
                if col_name not in normalized_columns:
                    normalized_columns.append(col_name)

    else:
        # normalize globally
        for col in metric_columns:
            if col not in df.columns:
                continue

            values = df[col].values.astype(float)

            if method == "standardize":
                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val > 0:
                    normalized = (values - mean_val) / std_val
                else:
                    normalized = values * 0
            elif method == "minmax":
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val > min_val:
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    normalized = values * 0
            elif method == "robust":
                median_val = np.median(values)
                q75 = np.percentile(values, 75)
                q25 = np.percentile(values, 25)
                iqr = q75 - q25
                if iqr > 0:
                    normalized = (values - median_val) / iqr
                else:
                    normalized = values * 0
            else:
                raise ValueError(
                    f"Unknown method: {method}. "
                    "Must be one of: standardize, minmax, robust"
                )

            col_name = f"{col}_normalized"
            df[col_name] = normalized
            normalized_columns.append(col_name)

    return df


def load_and_preprocess(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    target_timezone: Optional[str] = "UTC",
    resample_frequency: Optional[str] = None,
    resample_aggregation: str = "sum",
    handle_missing: bool = True,
    missing_method: str = "forward",
    normalize: bool = False,
    normalization_method: str = "standardize",
    normalize_per_id: bool = True,
) -> pd.DataFrame:
    """
    Load and preprocess time series data in one step.

    Parameters
    ----------
    file_path : str or Path
        Path to the data file
    file_format : str, optional
        File format: 'parquet', 'csv', or None for auto-detection
    target_timezone : str, optional
        Target timezone for timestamps (default: UTC)
    resample_frequency : str, optional
        Target frequency for resampling (e.g., 'h', 'D'). If None, no resampling
    resample_aggregation : str
        Aggregation method for resampling: 'sum', 'mean', 'max', 'min', 'last'
    handle_missing : bool
        Whether to handle missing values
    missing_method : str
        Method for handling missing values: 'forward', 'backward', 'zero', 'interpolate', 'drop'
    normalize : bool
        Whether to normalize metrics
    normalization_method : str
        Normalization method: 'standardize', 'minmax', 'robust'
    normalize_per_id : bool
        Whether to normalize separately per ID (video)

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for EDA and feature engineering
    """
    from src.data.load_data import load_data

    # load data
    df = load_data(file_path, file_format=file_format)

    # handle timezone
    df = handle_timezone(df, target_timezone=target_timezone)

    # sort by id and timestamp
    df = sort_by_id_and_timestamp(df)

    # resample if requested
    if resample_frequency is not None:
        df = resample_timeseries(
            df,
            frequency=resample_frequency,
            aggregation_method=resample_aggregation,
        )

    # handle missing values
    if handle_missing:
        df = handle_missing_values(df, method=missing_method)

    # normalize if requested
    if normalize:
        df = normalize_metrics(
            df,
            method=normalization_method,
            group_by_id=normalize_per_id,
        )

    return df

