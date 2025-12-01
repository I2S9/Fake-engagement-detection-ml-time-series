"""
Configuration utilities for loading and managing config files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default config/config.yaml

    Returns
    -------
    dict
        Configuration dictionary
    """
    if config_path is None:
        # default path
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def update_config_with_data(config: Dict[str, Any], input_size: int, seq_len: int) -> Dict[str, Any]:
    """
    Update configuration with data-specific parameters.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    input_size : int
        Number of input features
    seq_len : int
        Sequence length

    Returns
    -------
    dict
        Updated configuration dictionary
    """
    config = config.copy()

    # update LSTM config
    if "lstm" in config:
        config["lstm"]["input_size"] = input_size

    # update TCN config
    if "tcn" in config:
        config["tcn"]["input_size"] = input_size

    # update Autoencoder config
    if "autoencoder" in config:
        config["autoencoder"]["input_size"] = input_size
        config["autoencoder"]["seq_len"] = seq_len

    return config

