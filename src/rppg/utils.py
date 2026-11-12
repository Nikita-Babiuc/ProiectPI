#Small helpers shared by the pipeline and the scripts

from __future__ import annotations

import logging
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union

import yaml

T = TypeVar("T")


# Reads a YAML file into a dict, returns an empty dict for an empty file.
def load_config(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level")
    return data


# Builds a dataclass from a dict, rejecting unknown keys so typos in the YAML do not pass silently.
def build_dataclass(cls: Type[T], data: Optional[dict[str, Any]]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    data = dict(data or {})
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {unknown}. Valid: {sorted(known)}")
    for name, value in data.items():
        # YAML gives lists, so tuple fields are converted back.
        if isinstance(value, list) and "tuple" in str(known[name].type):
            data[name] = tuple(value)
    return cls(**data)


# Configures the root logger with a short format.
def setup_logging(level: Union[int, str] = "INFO") -> None:
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


# Measures the wall time of a with block in seconds.
class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0
        self._t0 = 0.0

    # Starts the clock.
    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    # Stops the clock and logs the result at debug level.
    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0
        if self.name:
            logging.getLogger(__name__).debug("%s took %.2f ms", self.name, self.elapsed * 1000.0)


# Keeps a smoothed frames-per-second value for on-screen display.
class FpsMeter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last: Optional[float] = None

    # Call once per frame, returns the smoothed fps.
    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0.0 else self.smoothing * self.fps + (1.0 - self.smoothing) * inst
        self._last = now
        return self.fps