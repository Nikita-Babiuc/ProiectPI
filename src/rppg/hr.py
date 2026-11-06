#Estimates heart rate from a filtered pulse signal: spectrum peak to bpm, signal-to-noise quality,
# and temporal smoothing of the estimates.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

_TINY = np.finfo(np.float64).tiny


@dataclass
class HrConfig:
    low_hz: float = 0.7
    high_hz: float = 3.0
    segment_seconds: float = 10.0
    resolution_hz: float = 0.01
    peak_halfwidth_hz: float = 0.2
    min_seconds: float = 5.0
    min_snr_db: float = 3.0


@dataclass(frozen=True)
class HrEstimate:
    bpm: float
    freq_hz: float
    snr_db: float
    valid: bool


@dataclass
class TrackerConfig:
    smoothing_seconds: float = 3.0
    max_change_bpm_per_s: float = 8.0
    hold_seconds: float = 5.0


# Computes the Welch power spectrum, zero-padded so the frequency grid is finer than resolution_hz.
def power_spectrum(
    x: np.ndarray,
    fs: float,
    segment_seconds: float = 10.0,
    resolution_hz: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if x.size < 8:
        raise ValueError("signal is too short for a spectrum")

    nperseg = min(x.size, int(round(segment_seconds * fs)))
    nfft = max(nperseg, int(2 ** math.ceil(math.log2(fs / resolution_hz))))
    return signal.welch(
        x, fs=fs, window="hann", nperseg=nperseg, nfft=nfft, detrend="constant"
    )


# Estimates the heart rate of a pulse signal from its spectrum peak, with an SNR quality measure.
def estimate_hr(
    pulse: np.ndarray, fs: float, config: Optional[HrConfig] = None
) -> HrEstimate:
    cfg = config or HrConfig()
    if fs <= 0:
        raise ValueError("fs must be positive")

    pulse = np.asarray(pulse, dtype=np.float64)
    invalid = HrEstimate(float("nan"), float("nan"), float("nan"), False)
    if pulse.size < max(int(cfg.min_seconds * fs), 8) or not np.isfinite(pulse).all():
        return invalid

    freqs, psd = power_spectrum(pulse, fs, cfg.segment_seconds, cfg.resolution_hz)
    band = (freqs >= cfg.low_hz) & (freqs <= cfg.high_hz)
    if not band.any() or not np.any(psd[band] > 0):
        return invalid

    peak = int(np.argmax(np.where(band, psd, -np.inf)))
    freq = float(freqs[peak])
    snr = _snr_db(freqs, psd, freq, cfg)
    return HrEstimate(60.0 * freq, freq, snr, snr >= cfg.min_snr_db)


# Returns the ratio in dB between the power around the peak and its first harmonic and the rest of the band.
def _snr_db(freqs: np.ndarray, psd: np.ndarray, peak_hz: float, cfg: HrConfig) -> float:
    band = (freqs >= cfg.low_hz) & (freqs <= cfg.high_hz)
    near = (np.abs(freqs - peak_hz) <= cfg.peak_halfwidth_hz) | (
        np.abs(freqs - 2.0 * peak_hz) <= cfg.peak_halfwidth_hz
    )
    signal_power = psd[band & near].sum()
    noise_power = psd[band & ~near].sum()
    return float(10.0 * np.log10(max(signal_power, _TINY) / max(noise_power, _TINY)))


class HrTracker:
    # Stores the config and starts with no heart rate.
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.cfg = config or TrackerConfig()
        self._bpm: Optional[float] = None
        self._last_valid_t: Optional[float] = None

    # Feeds one estimate taken at the given time and returns the smoothed bpm, or None if there is no reliable value.
    def update(self, timestamp: float, estimate: HrEstimate) -> Optional[float]:
        if estimate.valid:
            if self._bpm is None:
                self._bpm = estimate.bpm
            else:
                dt = max(timestamp - self._last_valid_t, 1e-3)
                limit = self.cfg.max_change_bpm_per_s * dt
                step = float(np.clip(estimate.bpm - self._bpm, -limit, limit))
                alpha = 1.0 - math.exp(-dt / self.cfg.smoothing_seconds)
                self._bpm += alpha * step
            self._last_valid_t = timestamp
        elif (
            self._bpm is not None
            and timestamp - self._last_valid_t > self.cfg.hold_seconds
        ):
            self._bpm = None
        return self._bpm

    # Returns the current smoothed bpm, or None.
    @property
    def bpm(self) -> Optional[float]:
        return self._bpm

    # Forgets the current heart rate.
    def reset(self) -> None:
        self._bpm = None
        self._last_valid_t = None