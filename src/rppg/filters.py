#Signal conditioning for rPPG: resamples irregular webcam samples to a uniform rate, removes the trend and
# applies a zero-phase Butterworth bandpass.

from __future__ import annotations

import numpy as np
from scipy import signal

DEFAULT_LOW_HZ = 0.7
DEFAULT_HIGH_HZ = 3.0
DEFAULT_ORDER = 3


# Estimates the sampling rate in Hz from the median time step between samples.
def estimate_fs(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float64)
    if t.size < 2:
        raise ValueError("need at least 2 timestamps to estimate fs")
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError("timestamps must be increasing")
    return float(1.0 / dt)


# Linearly interpolates samples taken at irregular times onto a uniform grid at fs Hz, works on (N,) or (N, C) data.
def resample_uniform(
    t: np.ndarray, x: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be 1D with at least 2 samples")
    if x.shape[0] != t.size:
        raise ValueError("x and t must have the same length")
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    n = int(np.floor((t[-1] - t[0]) * fs + 1e-9)) + 1
    grid = t[0] + np.arange(n) / fs

    if x.ndim == 1:
        return grid, np.interp(grid, t, x)
    out = np.column_stack([np.interp(grid, t, x[:, c]) for c in range(x.shape[1])])
    return grid, out


# Removes the best-fit straight line from the signal along the first axis.
def detrend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return x - x.mean(axis=0) if x.shape[0] else x
    return signal.detrend(x, axis=0, type="linear")


# Applies a zero-phase Butterworth bandpass, returning zeros if the signal is too short to filter.
def bandpass(
    x: np.ndarray,
    fs: float,
    low: float = DEFAULT_LOW_HZ,
    high: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nyquist = fs / 2.0
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not 0 < low < high < nyquist:
        raise ValueError(
            f"band ({low}, {high}) Hz must satisfy 0 < low < high < fs/2 = {nyquist:.2f}"
        )

    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    min_length = 3 * (2 * sos.shape[0] + 1) + 1
    if x.shape[0] < min_length:
        return np.zeros_like(x)
    return signal.sosfiltfilt(sos, x, axis=0)


# Detrends and bandpasses a pulse signal in one step.
def clean_pulse(
    x: np.ndarray,
    fs: float,
    low: float = DEFAULT_LOW_HZ,
    high: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    return bandpass(detrend(x), fs, low, high, order)