"""
Load raw time series data from files.

This module provides functions to load engagement time series data
from various file formats (parquet, CSV) and return clean DataFrames.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union
import os


def load_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a parquet file.

    Parameters
    ----------
    file_path : str or Path
        Path to the parquet file

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file cannot be read as parquet
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error reading parquet file {file_path}: {str(e)}")


def load_csv(
    file_path: Union[str, Path],
    parse_dates: Optional[list] = None,
    date_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file
    parse_dates : list, optional
        List of column names to parse as dates. If None, tries to parse 'timestamp'
    date_column : str
        Name of the timestamp column to parse

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file cannot be read as CSV
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if parse_dates is None:
            parse_dates = [date_column]

        df = pd.read_csv(file_path, parse_dates=parse_dates)
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file {file_path}: {str(e)}")


def load_data(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load time series data from a file (auto-detects format or uses specified format).

    Parameters
    ----------
    file_path : str or Path
        Path to the data file
    file_format : str, optional
        File format: 'parquet', 'csv', or None for auto-detection

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with columns: id, timestamp, views, likes, comments, shares, label

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file format is not supported or cannot be determined
    """
    file_path = Path(file_path)

    if file_format is None:
        # auto-detect format from extension
        suffix = file_path.suffix.lower()
        if suffix == ".parquet":
            file_format = "parquet"
        elif suffix in [".csv", ".tsv"]:
            file_format = "csv"
        else:
            raise ValueError(
                f"Cannot auto-detect format for {file_path}. "
                "Please specify file_format ('parquet' or 'csv')"
            )

    if file_format.lower() == "parquet":
        df = load_parquet(file_path)
    elif file_format.lower() == "csv":
        df = load_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    # validate required columns
    required_columns = ["id", "timestamp", "views", "likes", "comments", "shares", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Found columns: {df.columns.tolist()}"
        )

    return df


def load_from_directory(
    directory_path: Union[str, Path],
    file_pattern: str = "*.parquet",
) -> pd.DataFrame:
    """
    Load all matching files from a directory and concatenate them.

    Parameters
    ----------
    directory_path : str or Path
        Path to the directory containing data files
    file_pattern : str
        Glob pattern to match files (e.g., "*.parquet", "*.csv")

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all matching files

    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    ValueError
        If no files match the pattern
    """
    directory_path = Path(directory_path)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    files = list(directory_path.glob(file_pattern))
    if not files:
        raise ValueError(
            f"No files found matching pattern '{file_pattern}' in {directory_path}"
        )

    dataframes = []
    for file_path in files:
        try:
            df = load_data(file_path)
            dataframes.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {str(e)}")
            continue

    if not dataframes:
        raise ValueError(f"Could not load any files from {directory_path}")

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

