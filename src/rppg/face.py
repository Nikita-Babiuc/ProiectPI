#Detects the face in each frame and returns its landmarks, using MediaPipe FaceLandmarker.

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .capture import Frame

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "rppg" / "face_landmarker.task"


@dataclass(frozen=True)
class FaceResult:

    landmarks_px: np.ndarray
    landmarks_norm: np.ndarray
    bbox: tuple[int, int, int, int]
    timestamp: float
    index: int


# Returns the model path, downloading the model first if it is missing.
def ensure_model(path: Optional[Path] = None, url: str = MODEL_URL) -> Path:
    path = Path(path) if path else DEFAULT_MODEL_PATH
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading face landmarker model to %s", path)
    tmp = path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the face landmarker model from {url}. "
            f"Download it manually and place it at {path}."
        ) from e
    return path


class FaceTracker:

    # Loads the model and creates the MediaPipe FaceLandmarker.
    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth: float = 0.5,
    ):
        if not 0.0 <= smooth < 1.0:
            raise ValueError("smooth must be in [0, 1)")
        self.smooth = smooth

        model = ensure_model(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        self._prev_px: Optional[np.ndarray] = None

        self.frames_seen = 0
        self.frames_detected = 0

    # Closes the landmarker.
    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    # Returns the tracker when entering a with block.
    def __enter__(self) -> "FaceTracker":
        return self

    # Closes the tracker when leaving a with block.
    def __exit__(self, *exc) -> None:
        self.close()

    # Detects and tracks the face in one frame and returns its landmarks, or None if no face is found.
    def process(self, frame: Frame) -> Optional[FaceResult]:
        h, w = frame.image.shape[:2]
        self.frames_seen += 1

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = max(int(frame.timestamp * 1000.0), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            self._prev_px = None
            return None

        lm = result.face_landmarks[0]
        norm = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32)
        px = norm[:, :2] * np.array([w, h], dtype=np.float32)

        if self.smooth > 0.0:
            if self._prev_px is not None and self._prev_px.shape == px.shape:
                px = self.smooth * self._prev_px + (1.0 - self.smooth) * px
            self._prev_px = px

        self.frames_detected += 1
        return FaceResult(
            landmarks_px=px.astype(np.float32),
            landmarks_norm=norm,
            bbox=_bbox_from_points(px, w, h),
            timestamp=frame.timestamp,
            index=frame.index,
        )

    # Returns the fraction of frames in which a face was found.
    @property
    def detection_rate(self) -> float:
        return self.frames_detected / self.frames_seen if self.frames_seen else 0.0


# Computes a bounding box around the landmarks, clipped to the image.
def _bbox_from_points(px: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = np.floor(px.min(axis=0)).astype(int)
    x1, y1 = np.ceil(px.max(axis=0)).astype(int)
    return (
        int(np.clip(x0, 0, w - 1)),
        int(np.clip(y0, 0, h - 1)),
        int(np.clip(x1, 0, w - 1)),
        int(np.clip(y1, 0, h - 1)),
    )


# Draws the bounding box and the landmarks on the image
def draw_face(image: np.ndarray, face: Optional[FaceResult], draw_points: bool = True) -> np.ndarray:
    if face is None:
        return image
    x0, y0, x1, y1 = face.bbox
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    if draw_points:
        for x, y in face.landmarks_px:
            cv2.circle(image, (int(x), int(y)), 1, (0, 200, 255), -1)
    return image


if __name__ == "__main__":
    import sys

    from .capture import open_source

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    src = open_source(int(arg) if arg.isdigit() else arg)

    t_prev = time.perf_counter()
    fps = 0.0
    with src, FaceTracker() as tracker:
        for frame in src:
            face = tracker.process(frame)
            vis = draw_face(frame.image.copy(), face)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            status = "face" if face is not None else "NO FACE"
            cv2.putText(
                vis, f"{status}  proc fps={fps:4.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if face is not None else (0, 0, 255), 2,
            )
            cv2.imshow("face", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    print(f"detection rate: {tracker.detection_rate:.1%} "
          f"({tracker.frames_detected}/{tracker.frames_seen})")