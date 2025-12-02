"""
Realistic behavior simulation for engagement time series.

This module generates time series with multiple user profiles and attack types,
incorporating human natural cycles and realistic patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import random

# Global RNG for reproducibility
RNG = np.random.default_rng(seed=42)


def generate_base_pattern(
    length: int,
    daily_cycle: bool = True,
    weekly_cycle: bool = True,
    seasonal_cycle: bool = True,
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> np.ndarray:
    """
    Generate base pattern with human natural cycles.

    Parameters
    ----------
    length : int
        Length of the time series
    daily_cycle : bool
        Whether to include daily cycle
    weekly_cycle : bool
        Whether to include weekly cycle
    seasonal_cycle : bool
        Whether to include seasonal cycle
    timestamps : pd.DatetimeIndex, optional
        Timestamps for the series (needed for realistic cycles)

    Returns
    -------
    np.ndarray
        Base pattern array
    """
    t = np.arange(length)
    base = np.ones(length, dtype=float)

    if timestamps is not None and daily_cycle:
        # realistic day/night cycle
        hour_of_day = np.array([d.hour for d in timestamps])
        day_night = np.ones(length)
        
        for i, hour in enumerate(hour_of_day):
            if 0 <= hour < 6:
                # Night: very low
                day_night[i] = 0.15 + 0.20 * RNG.random()
            elif 6 <= hour < 9:
                # Morning: rising
                progress = (hour - 6) / 3.0
                day_night[i] = 0.4 + 0.4 * progress + 0.1 * RNG.random()
            elif 9 <= hour < 18:
                # Day: moderate
                day_night[i] = 0.7 + 0.4 * RNG.random()
            elif 18 <= hour < 23:
                # Evening: peak (pics après 18h)
                peak = 1.2 + 0.3 * RNG.random()
                if 20 <= hour <= 21:
                    peak += 0.2 + 0.1 * RNG.random()
                day_night[i] = peak
            else:  # 23-0h
                day_night[i] = 0.5 + 0.2 * RNG.random()
        
        base *= day_night

    if timestamps is not None and weekly_cycle:
        # realistic weekly cycle
        day_of_week = np.array([d.weekday() for d in timestamps])
        weekly = np.ones(length)
        
        for i, weekday in enumerate(day_of_week):
            if weekday == 0:  # Monday
                weekly[i] = 0.9 + 0.2 * RNG.random()
            elif weekday in [1, 2, 3]:  # Tue-Thu
                weekly[i] = 1.0 + 0.2 * RNG.random()
            elif weekday == 4:  # Friday
                weekly[i] = 1.1 + 0.2 * RNG.random()
            elif weekday == 5:  # Saturday
                weekly[i] = 1.4 + 0.3 * RNG.random()
            else:  # Sunday
                weekly[i] = 1.2 + 0.3 * RNG.random()
        
        base *= weekly
        
        # additional weekend boost
        is_weekend = (day_of_week >= 5)
        base[is_weekend] *= 1.15

    if timestamps is not None and seasonal_cycle:
        # weak seasonality
        day_of_year = np.array([d.timetuple().tm_yday for d in timestamps])
        seasonal = 1.0 + 0.1 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        seasonal = np.clip(seasonal, 0.9, 1.1)
        base *= seasonal

    # add realistic noise
    noise = RNG.normal(loc=0.0, scale=0.1, size=length)
    base += noise
    base = np.clip(base, 0.0, None)

    return base


def simulate_normal_user(
    length: int,
    profile: str = "regular",
    timestamps: Optional[pd.DatetimeIndex] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a normal user engagement pattern.

    Parameters
    ----------
    length : int
        Length of the time series
    profile : str
        User profile type: 'regular', 'impulsive', 'dormant', 'influencer', 'new'
    timestamps : pd.DatetimeIndex, optional
        Timestamps for realistic cycles
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple
        (views, likes, comments) arrays
    """
    if random_seed is not None:
        local_rng = np.random.default_rng(random_seed)
    else:
        local_rng = RNG

    base = generate_base_pattern(length, timestamps=timestamps)

    if profile == "regular":
        # typical user with consistent engagement
        scale = 20
        base *= local_rng.uniform(0.8, 1.2, length)

    elif profile == "impulsive":
        # user with occasional spikes (organic viral moments)
        scale = 30
        base *= local_rng.uniform(0.9, 1.3, length)
        # add 2-4 organic spikes
        n_spikes = local_rng.integers(2, 5)
        spikes_idx = local_rng.choice(length, size=n_spikes, replace=False)
        for idx in spikes_idx:
            base[idx] *= local_rng.uniform(1.5, 2.5)

    elif profile == "dormant":
        # low activity user
        scale = 5
        base *= local_rng.uniform(0.5, 0.8, length)

    elif profile == "influencer":
        # high baseline with occasional peaks
        scale = 100
        base *= local_rng.uniform(1.2, 1.8, length)
        # occasional viral moments
        n_viral = local_rng.integers(1, 3)
        viral_idx = local_rng.choice(length, size=n_viral, replace=False)
        for idx in viral_idx:
            base[idx] *= local_rng.uniform(2.0, 3.5)

    elif profile == "new":
        # new user with growing trend
        scale = 8
        trend = np.linspace(0.5, 1.5, length)
        base *= trend
        base *= local_rng.uniform(0.7, 1.1, length)

    elif profile == "casual":
        # casual viewer, moderate activity
        scale = 15
        base *= local_rng.uniform(0.7, 1.0, length)

    elif profile == "power":
        # power user, high consistent activity
        scale = 50
        base *= local_rng.uniform(1.1, 1.4, length)

    else:
        scale = 20

    views = np.clip(np.round(base * scale), 0, None).astype(int)

    # realistic engagement ratios
    like_ratio = local_rng.uniform(0.03, 0.12)
    comment_ratio = local_rng.uniform(0.005, 0.02)
    
    likes = np.clip(
        np.round(views * like_ratio * local_rng.uniform(0.8, 1.2, length)),
        0,
        None,
    ).astype(int)
    
    comments = np.clip(
        np.round(views * comment_ratio * local_rng.uniform(0.7, 1.3, length)),
        0,
        None,
    ).astype(int)

    # ensure logical constraints
    likes = np.minimum(likes, views)
    comments = np.minimum(comments, views)

    return views, likes, comments


def apply_fake_pattern(
    views: np.ndarray,
    attack_type: str = "boost_progressive",
    timestamps: Optional[pd.DatetimeIndex] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply fake engagement pattern to views.

    Parameters
    ----------
    views : np.ndarray
        Original views array
    attack_type : str
        Type of attack: 'boost_progressive', 'bursts_small', 'wave_spam',
        'overlay_fake_on_trend', 'single_spike', 'off_peak_bursts'
    timestamps : pd.DatetimeIndex, optional
        Timestamps for time-aware attacks
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple
        (views_fake, anomaly_mask) where anomaly_mask indicates anomalous windows
    """
    if random_seed is not None:
        local_rng = np.random.default_rng(random_seed)
    else:
        local_rng = RNG

    length = len(views)
    anomaly_mask = np.zeros(length, dtype=bool)
    views_fake = views.astype(float).copy()

    if attack_type == "boost_progressive":
        # gradual increase to make it look organic
        start = local_rng.integers(low=length // 4, high=length // 2)
        factor = np.linspace(1.2, 3.0, length - start)
        views_fake[start:] *= factor
        anomaly_mask[start:] = True

    elif attack_type == "bursts_small":
        # multiple small bursts
        n_bursts = local_rng.integers(3, 7)
        idx = local_rng.choice(length, size=n_bursts, replace=False)
        for i in idx:
            burst_size = local_rng.uniform(2.0, 4.0)
            views_fake[i] *= burst_size
            anomaly_mask[i] = True
            # small spillover effect
            if i + 1 < length:
                views_fake[i + 1] *= 1.2
                anomaly_mask[i + 1] = True

    elif attack_type == "wave_spam":
        # regular wave pattern (bot-like)
        period = local_rng.integers(6, 12)
        for i in range(0, length, period):
            end = min(i + 3, length)
            multiplier = local_rng.uniform(2.5, 3.5)
            views_fake[i:end] *= multiplier
            anomaly_mask[i:end] = True

    elif attack_type == "overlay_fake_on_trend":
        # fake engagement overlaid on natural trend
        start = local_rng.integers(low=length // 3, high=length // 2)
        end = min(start + local_rng.integers(6, 24), length)
        trend = np.linspace(1.0, 4.0, end - start)
        views_fake[start:end] *= trend
        anomaly_mask[start:end] = True

    elif attack_type == "single_spike":
        # single massive spike (obvious fake)
        i = local_rng.integers(low=length // 4, high=3 * length // 4)
        views_fake[i] *= local_rng.uniform(5.0, 8.0)
        anomaly_mask[i] = True

    elif attack_type == "off_peak_bursts":
        # bursts during off-peak hours (suspicious)
        if timestamps is not None:
            hour_of_day = np.array([d.hour for d in timestamps])
            off_peak_mask = (hour_of_day >= 2) & (hour_of_day <= 5)
            off_peak_indices = np.where(off_peak_mask)[0]
            if len(off_peak_indices) > 0:
                n_bursts = min(local_rng.integers(2, 5), len(off_peak_indices))
                burst_idx = local_rng.choice(off_peak_indices, size=n_bursts, replace=False)
                for idx in burst_idx:
                    views_fake[idx] *= local_rng.uniform(3.0, 6.0)
                    anomaly_mask[idx] = True
        else:
            # fallback: random bursts
            n_bursts = local_rng.integers(2, 5)
            idx = local_rng.choice(length, size=n_bursts, replace=False)
            for i in idx:
                views_fake[i] *= local_rng.uniform(3.0, 6.0)
                anomaly_mask[i] = True

    elif attack_type == "perfect_sync":
        # perfect synchronization (unrealistic correlation)
        sync_signal = 1.0 + 0.8 * np.sin(2 * np.pi * np.arange(length) / 12)
        views_fake *= sync_signal
        anomaly_mask[:] = True

    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")

    views_fake = np.clip(np.round(views_fake), 0, None).astype(int)

    return views_fake, anomaly_mask


def simulate_user_series(
    user_id: str,
    start_timestamp: datetime,
    length: int,
    freq: str = "H",
    profile: str = "regular",
    is_fake: bool = False,
    attack_type: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate a complete user engagement series.

    Parameters
    ----------
    user_id : str
        Unique user identifier
    start_timestamp : datetime
        Start timestamp for the series
    length : int
        Length of the series (in time units)
    freq : str
        Frequency: 'H' for hourly, 'D' for daily
    profile : str
        User profile type
    is_fake : bool
        Whether this is a fake engagement series
    attack_type : str, optional
        Type of fake attack (required if is_fake=True)
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: user_id, timestamp, views, likes, comments,
        is_fake_series, is_anomaly_window, profile, attack_type
    """
    if random_seed is not None:
        local_rng = np.random.default_rng(random_seed)
    else:
        local_rng = RNG

    # generate timestamps (normalize freq to lowercase)
    freq_normalized = freq.lower() if freq == "H" else freq
    timestamps = pd.date_range(start=start_timestamp, periods=length, freq=freq_normalized)

    # simulate normal user
    views, likes, comments = simulate_normal_user(
        length, profile=profile, timestamps=timestamps, random_seed=random_seed
    )

    anomaly_mask = np.zeros(length, dtype=bool)

    # apply fake pattern if needed
    if is_fake and attack_type is not None:
        views, anomaly_mask = apply_fake_pattern(
            views, attack_type=attack_type, timestamps=timestamps, random_seed=random_seed
        )

        # recalculate engagement metrics based on fake views
        like_ratio = local_rng.uniform(0.03, 0.12)
        comment_ratio = local_rng.uniform(0.005, 0.02)

        likes = np.clip(
            np.round(views * like_ratio * local_rng.uniform(0.8, 1.2, length)),
            0,
            None,
        ).astype(int)

        comments = np.clip(
            np.round(views * comment_ratio * local_rng.uniform(0.7, 1.3, length)),
            0,
            None,
        ).astype(int)

        # ensure logical constraints
        likes = np.minimum(likes, views)
        comments = np.minimum(comments, views)

    # create shares (derived from views)
    shares = np.clip(
        np.round(views * local_rng.uniform(0.01, 0.03, length)),
        0,
        None,
    ).astype(int)
    shares = np.minimum(shares, views)

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "timestamp": timestamps,
            "views": views,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "is_fake_series": is_fake,
            "is_anomaly_window": anomaly_mask,
            "profile": profile,
            "attack_type": attack_type if is_fake else "none",
        }
    )

    return df


def generate_dataset(
    n_users: int = 1000,
    length: int = 24 * 7,  # 1 week hourly
    fake_ratio: float = 0.3,
    start_timestamp: str = "2024-01-01",
    freq: str = "H",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a complete dataset with multiple user profiles and attack types.

    Parameters
    ----------
    n_users : int
        Number of users to generate
    length : int
        Length of each series (in time units)
    fake_ratio : float
        Proportion of fake engagement series
    start_timestamp : str
        Start timestamp (will be varied per user)
    freq : str
        Frequency: 'H' for hourly, 'D' for daily
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Combined dataset with all user series
    """
    global RNG
    RNG = np.random.default_rng(seed=random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    profiles = ["regular", "impulsive", "dormant", "influencer", "new", "casual", "power"]
    attack_types = [
        "boost_progressive",
        "bursts_small",
        "wave_spam",
        "overlay_fake_on_trend",
        "single_spike",
        "off_peak_bursts",
        "perfect_sync",
    ]

    all_series = []
    start_dt = pd.to_datetime(start_timestamp)

    for user_idx in range(n_users):
        user_id = f"user_{user_idx+1:04d}"

        # determine if fake
        is_fake = RNG.random() < fake_ratio

        # select profile
        profile = RNG.choice(profiles)

        # select attack type if fake
        if is_fake:
            attack_type = RNG.choice(attack_types)
        else:
            attack_type = None

        # vary start timestamp slightly
        hours_offset = int(RNG.integers(0, 24)) if freq == "H" else 0
        days_offset = int(RNG.integers(0, 7)) if freq == "D" else 0
        user_start = start_dt + timedelta(hours=hours_offset, days=days_offset)

        df_user = simulate_user_series(
            user_id=user_id,
            start_timestamp=user_start,
            length=length,
            freq=freq,
            profile=profile,
            is_fake=is_fake,
            attack_type=attack_type,
            random_seed=random_seed + user_idx,
        )

        all_series.append(df_user)

    full_df = pd.concat(all_series, ignore_index=True)

    return full_df

