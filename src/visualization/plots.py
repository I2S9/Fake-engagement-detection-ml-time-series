# src/visualization/plots.py

import numpy as np
import matplotlib.pyplot as plt


def plot_series_with_anomalies(timestamps, values, anomaly_mask, title=None):
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(timestamps, values, label="series", linewidth=1.2)

    if anomaly_mask is not None and anomaly_mask.any():
        ax.scatter(
            np.array(timestamps)[anomaly_mask],
            np.array(values)[anomaly_mask],
            marker="o",
            s=20,
            label="anomaly",
        )

    ax.set_xlabel("time")
    ax.set_ylabel("value")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_score_with_threshold(timestamps, scores, threshold, title=None):
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(timestamps, scores, label="anomaly score", linewidth=1.2)
    ax.axhline(threshold, linestyle="--", label=f"threshold {threshold:.2f}")

    ax.set_xlabel("time")
    ax.set_ylabel("score")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_reconstruction(original, reconstructed, anomaly_mask=None, title=None):
    t = np.arange(len(original))

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t, original, label="original", linewidth=1.2)
    ax.plot(t, reconstructed, label="reconstruction", linewidth=1.2)

    if anomaly_mask is not None and anomaly_mask.any():
        ax.scatter(
            t[anomaly_mask],
            np.array(original)[anomaly_mask],
            marker="o",
            s=20,
            label="high error",
        )

    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax

