#POS  algorithm: converts a uniformly sampled RGB trace into a pulse signal, using NumPy only.

from __future__ import annotations

import math

import numpy as np

PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
_EPS = 1e-12


# Computes the POS pulse signal from an (N, 3) RGB trace sampled uniformly at fs Hz.
def pos(rgb: np.ndarray, fs: float, window_seconds: float = 1.6) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape (N, 3)")
    if fs <= 0:
        raise ValueError("fs must be positive")

    n = rgb.shape[0]
    length = math.ceil(window_seconds * fs)
    if length < 2:
        raise ValueError("window is too short, increase window_seconds or fs")

    pulse = np.zeros(n, dtype=np.float64)
    if n < length:
        return pulse

    for start, h in enumerate(_window_pulses(rgb, length)):
        pulse[start : start + length] += h
    return pulse


# Returns one zero-mean pulse segment of the given length for every sliding window
def _window_pulses(rgb: np.ndarray, length: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(rgb, length, axis=0)
    normed = windows / np.maximum(windows.mean(axis=2, keepdims=True), _EPS)

    s = np.einsum("pc,wcl->wpl", PROJECTION, normed)
    s1, s2 = s[:, 0, :], s[:, 1, :]

    std1, std2 = s1.std(axis=1), s2.std(axis=1)
    alpha = np.divide(std1, std2, out=np.zeros_like(std1), where=std2 > _EPS)

    h = s1 + alpha[:, None] * s2
    return h - h.mean(axis=1, keepdims=True)