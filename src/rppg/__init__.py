# Public API of the package. Names are imported on first use, so importing a light module
# (filters, pos, hr) does not load OpenCV or MediaPipe.

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

_EXPORTS = {
    "CameraConfig": "capture",
    "Frame": "capture",
    "VideoFileCapture": "capture",
    "WebcamCapture": "capture",
    "open_source": "capture",
    "FaceTracker": "face",
    "RoiConfig": "roi",
    "RoiExtractor": "roi",
    "HrConfig": "hr",
    "TrackerConfig": "hr",
    "HrResult": "pipeline",
    "PipelineConfig": "pipeline",
    "RPPGPipeline": "pipeline",
    "load_config": "utils",
}

__all__ = ["__version__", *_EXPORTS]

if TYPE_CHECKING:
    from .capture import CameraConfig, Frame, VideoFileCapture, WebcamCapture, open_source
    from .face import FaceTracker
    from .hr import HrConfig, TrackerConfig
    from .pipeline import HrResult, PipelineConfig, RPPGPipeline
    from .roi import RoiConfig, RoiExtractor
    from .utils import load_config


# Imports the module that holds the requested name the first time it is used.
def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value