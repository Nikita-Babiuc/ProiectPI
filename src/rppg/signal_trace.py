#Turns each frame's ROI mask into one mean RGB sample and keeps a sliding buffer of
# recent samples with their timestamps.

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from .roi import RoiResult

log = logging.getLogger(__name__)


# Returns the spatial mean of the pixels inside the mask, in RGB order.
def mean_rgb(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b, g, r, _ = cv2.mean(image_bgr, mask=mask)
    return np.array([r, g, b], dtype=np.float64)


@dataclass
class TraceConfig:
    window_seconds: float = 10.0
    max_gap_seconds: Optional[float] = 1.0


class SignalTrace:
    # Stores the config and creates the empty buffers.
    def __init__(self, config: Optional[TraceConfig] = None):
        self.cfg = config or TraceConfig()
        self._t: deque[float] = deque()
        self._rgb: deque[np.ndarray] = deque()
        self.resets = 0
        self.skipped = 0

    # Adds one RGB sample, ignoring timestamps that do not move forward.
    def add(self, timestamp: float, rgb: np.ndarray) -> bool:
        if self._t and timestamp <= self._t[-1]:
            return False
        self._expire(timestamp)
        self._t.append(float(timestamp))
        self._rgb.append(np.asarray(rgb, dtype=np.float64))
        self._trim()
        return True

    # Computes the mean RGB of the ROI for one frame and adds it, returns False if the frame was skipped.
    def update(
        self, timestamp: float, image_bgr: np.ndarray, roi: Optional["RoiResult"]
    ) -> bool:
        if roi is None or not roi.valid:
            self._expire(timestamp)
            self.skipped += 1
            return False
        return self.add(timestamp, mean_rgb(image_bgr, roi.mask))

    # Clears the buffer if the newest sample is older than the allowed gap.
    def _expire(self, timestamp: float) -> None:
        gap = self.cfg.max_gap_seconds
        if self._t and gap is not None and timestamp - self._t[-1] > gap:
            log.debug("Gap of %.2fs, clearing trace buffer", timestamp - self._t[-1])
            self.clear()
            self.resets += 1

    # Drops the samples that fall outside the time window.
    def _trim(self) -> None:
        cutoff = self._t[-1] - self.cfg.window_seconds
        while self._t and self._t[0] < cutoff:
            self._t.popleft()
            self._rgb.popleft()

    # Returns the timestamps (N,) and the RGB samples (N, 3) currently in the buffer.
    def get(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._t:
            return np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64)
        t = np.fromiter(self._t, dtype=np.float64, count=len(self._t))
        return t, np.stack(self._rgb)

    # Empties the buffer.
    def clear(self) -> None:
        self._t.clear()
        self._rgb.clear()

    # Returns the number of samples in the buffer.
    def __len__(self) -> int:
        return len(self._t)

    # Returns the time span covered by the buffer in seconds.
    @property
    def duration(self) -> float:
        return self._t[-1] - self._t[0] if len(self._t) > 1 else 0.0

    # Returns the average sampling rate of the buffer in Hz.
    @property
    def fps(self) -> float:
        d = self.duration
        return (len(self._t) - 1) / d if d > 0 else 0.0


# Draws one channel of the trace as a simple line plot with the slow drift removed.
def draw_trace(
    values: np.ndarray,
    width: int = 480,
    height: int = 120,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    v = np.asarray(values, dtype=np.float64)[-width:]
    k = min(30, len(v) // 2)
    if k < 2:
        return canvas
    avg = np.convolve(v, np.ones(k) / k, mode="valid")
    v = v[k // 2 : k // 2 + len(avg)] - avg
    span = np.abs(v).max()
    if len(v) < 2 or span < 1e-9:
        return canvas
    x = np.linspace(0, width - 1, len(v))
    y = height / 2 - v / span * (height / 2 - 4)
    pts = np.stack([x, y], axis=1).astype(np.int32)
    cv2.polylines(canvas, [pts], False, color, 1)
    return canvas


if __name__ == "__main__":
    import sys

    from .capture import open_source
    from .face import FaceTracker
    from .roi import RoiConfig, RoiExtractor, draw_roi

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    src = open_source(int(arg) if arg.isdigit() else arg)
    extractor = RoiExtractor(RoiConfig())
    trace = SignalTrace()

    with src, FaceTracker() as tracker:
        for frame in src:
            vis = frame.image.copy()
            face = tracker.process(frame)
            roi = extractor.extract(frame.image, face) if face is not None else None
            trace.update(frame.timestamp, frame.image, roi)
            if roi is not None:
                draw_roi(vis, roi)

            text = (
                f"samples={len(trace)} fps={trace.fps:.1f} "
                f"span={trace.duration:.1f}s resets={trace.resets}"
            )
            cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("video", vis)
            if len(trace) > 1:
                cv2.imshow("green channel", draw_trace(trace.get()[1][:, 1]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()