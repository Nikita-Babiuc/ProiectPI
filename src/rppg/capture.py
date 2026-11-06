#Frame sources for the pipeline: reads frames from a webcam or a video file and gives each one a real timestamp.

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:

    image: np.ndarray
    timestamp: float
    index: int


@dataclass
class CameraConfig:
    index: int = 0
    width: Optional[int] = 640
    height: Optional[int] = 480
    fps: Optional[float] = 30.0
    fourcc: Optional[str] = "MJPG"
    backend: Optional[int] = None

    lock_exposure: bool = True
    lock_white_balance: bool = True
    lock_autofocus: bool = True
    exposure: Optional[float] = None
    wb_temperature: Optional[float] = None
    focus: Optional[float] = None

    threaded: bool = True
    queue_size: int = 64
    warmup_frames: int = 10
    max_consecutive_failures: int = 30


class FrameSource:

    nominal_fps: float = 0.0

    # Opens the source and returns it.
    def open(self) -> "FrameSource":
        raise NotImplementedError

    # Returns the next frame, or None when the stream ends.
    def read(self) -> Optional[Frame]:
        raise NotImplementedError

    # Closes the source.
    def release(self) -> None:
        raise NotImplementedError

    # Returns the frame size as (width, height).
    @property
    def size(self) -> tuple[int, int]:
        raise NotImplementedError

    # Opens the source when entering a with block.
    def __enter__(self) -> "FrameSource":
        return self.open()

    # Releases the source when leaving a with block.
    def __exit__(self, *exc) -> None:
        self.release()

    # Yields frames until the stream ends.
    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame


class WebcamCapture(FrameSource):

    # Stores the config and sets up the internal state and the frame queue.
    def __init__(self, config: Optional[CameraConfig] = None):
        self.cfg = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._t0: Optional[float] = None
        self._index = 0

        self._queue: "queue.Queue[Frame]" = queue.Queue(maxsize=self.cfg.queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._finished = threading.Event()

        self.dropped_frames = 0
        self.nominal_fps = 0.0

    # Opens the camera, applies the settings and locks, discards warmup frames and starts the capture thread.
    def open(self) -> "WebcamCapture":
        cfg = self.cfg
        if cfg.backend is not None:
            self._cap = cv2.VideoCapture(cfg.index, cfg.backend)
        else:
            self._cap = cv2.VideoCapture(cfg.index)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Could not open camera index {cfg.index}")

        self._configure_stream()
        self._apply_locks()

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(cfg.warmup_frames):
            self._cap.read()

        self.nominal_fps = self._cap.get(cv2.CAP_PROP_FPS) or (cfg.fps or 0.0)
        w, h = self.size
        log.info(
            "Camera %d opened: %dx%d, driver-reported fps=%.1f, backend=%s",
            cfg.index, w, h, self.nominal_fps, self._cap.getBackendName(),
        )

        self._t0 = None
        self._index = 0
        self.dropped_frames = 0
        self._stop.clear()
        self._finished.clear()

        if cfg.threaded:
            self._thread = threading.Thread(
                target=self._capture_loop, name="rppg-capture", daemon=True
            )
            self._thread.start()
        return self

    # Stops the capture thread and releases the camera.
    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # Returns the frame size as (width, height).
    @property
    def size(self) -> tuple[int, int]:
        assert self._cap is not None, "camera not open"
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    # Returns the next frame, from the queue in threaded mode or straight from the camera otherwise.
    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None
        if self.cfg.threaded:
            return self._read_threaded()
        return self._read_blocking()

    # Grabs a frame directly, retrying until it works or too many failures happen.
    def _read_blocking(self) -> Optional[Frame]:
        failures = 0
        while failures < self.cfg.max_consecutive_failures:
            frame = self._grab_one()
            if frame is not None:
                return frame
            failures += 1
        log.error("Camera stopped delivering frames")
        return None

    # Waits for the next frame from the queue, returns None when the capture has ended.
    def _read_threaded(self) -> Optional[Frame]:
        while True:
            try:
                return self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._finished.is_set() and self._queue.empty():
                    return None
                if self._stop.is_set():
                    return None

    # Background loop that keeps grabbing frames and pushing them to the queue.
    def _capture_loop(self) -> None:
        failures = 0
        try:
            while not self._stop.is_set():
                frame = self._grab_one()
                if frame is None:
                    failures += 1
                    if failures >= self.cfg.max_consecutive_failures:
                        log.error("Camera stopped delivering frames")
                        break
                    continue
                failures = 0
                self._enqueue(frame)
        finally:
            self._finished.set()

    # Puts a frame in the queue, dropping the oldest one if the queue is full.
    def _enqueue(self, frame: Frame) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self.dropped_frames += 1
            self._queue.put_nowait(frame)

    # Grabs one frame and timestamps it right after the grab, before decoding.
    def _grab_one(self) -> Optional[Frame]:
        assert self._cap is not None
        if not self._cap.grab():
            return None
        t = time.perf_counter()
        ok, image = self._cap.retrieve()
        if not ok or image is None:
            return None
        if self._t0 is None:
            self._t0 = t
        frame = Frame(image=image, timestamp=t - self._t0, index=self._index)
        self._index += 1
        return frame

    # Sets the codec, resolution and fps.
    def _configure_stream(self) -> None:
        cap, cfg = self._cap, self.cfg
        assert cap is not None
        if cfg.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        if cfg.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        if cfg.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        if cfg.fps:
            cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    # Tries to lock exposure, white balance and autofocus, and logs what the driver reports.
    def _apply_locks(self) -> None:
        cap, cfg = self._cap, self.cfg
        assert cap is not None
        backend = cap.getBackendName().upper()

        if cfg.lock_exposure:
            manual = 1 if "V4L" in backend else 0.25
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
            if cfg.exposure is not None:
                cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)
            log.info(
                "Exposure lock requested: auto_exposure=%s exposure=%s",
                cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), cap.get(cv2.CAP_PROP_EXPOSURE),
            )

        if cfg.lock_white_balance:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            if cfg.wb_temperature is not None:
                cap.set(cv2.CAP_PROP_WB_TEMPERATURE, cfg.wb_temperature)
            log.info(
                "White balance lock requested: auto_wb=%s temperature=%s",
                cap.get(cv2.CAP_PROP_AUTO_WB), cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
            )

        if cfg.lock_autofocus:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            if cfg.focus is not None:
                cap.set(cv2.CAP_PROP_FOCUS, cfg.focus)
            log.info(
                "Autofocus lock requested: autofocus=%s focus=%s",
                cap.get(cv2.CAP_PROP_AUTOFOCUS), cap.get(cv2.CAP_PROP_FOCUS),
            )


class VideoFileCapture(FrameSource):

    # Stores the file path and the timestamp options.
    def __init__(
        self,
        path: Union[str, Path],
        use_container_timestamps: bool = False,
        fps_override: Optional[float] = None,
    ):
        self.path = str(path)
        self.use_container_timestamps = use_container_timestamps
        self.fps_override = fps_override
        self._cap: Optional[cv2.VideoCapture] = None
        self._index = 0
        self.nominal_fps = 0.0
        self.frame_count = 0

    # Opens the video file and reads its fps and frame count.
    def open(self) -> "VideoFileCapture":
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Could not open video: {self.path}")

        self.nominal_fps = self.fps_override or self._cap.get(cv2.CAP_PROP_FPS)
        if not self.nominal_fps or self.nominal_fps <= 0:
            self.release()
            raise RuntimeError(
                f"Video reports no fps ({self.path}); pass fps_override=..."
            )
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._index = 0
        log.info(
            "Video opened: %s, %d frames @ %.2f fps",
            self.path, self.frame_count, self.nominal_fps,
        )
        return self

    # Releases the video file.
    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # Returns the frame size as (width, height).
    @property
    def size(self) -> tuple[int, int]:
        assert self._cap is not None, "video not open"
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    # Reads the next frame and assigns its timestamp.
    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        if self.use_container_timestamps:
            t = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        else:
            t = self._index / self.nominal_fps
        frame = Frame(image=image, timestamp=t, index=self._index)
        self._index += 1
        return frame


# Returns a webcam source for an int or a video file source for a path.
def open_source(source: Union[int, str, Path], **kwargs) -> FrameSource:
    if isinstance(source, int):
        return WebcamCapture(CameraConfig(index=source, **kwargs))
    return VideoFileCapture(source, **kwargs)


# Computes effective fps, jitter and the largest gap from a list of timestamps.
def timing_stats(timestamps: "np.ndarray | list[float]") -> dict:
    t = np.asarray(timestamps, dtype=np.float64)
    if t.size < 2:
        return {"n": int(t.size), "fps": 0.0, "dt_std_ms": 0.0, "max_gap_ms": 0.0}
    dt = np.diff(t)
    return {
        "n": int(t.size),
        "fps": float(1.0 / dt.mean()),
        "dt_std_ms": float(dt.std() * 1000.0),
        "max_gap_ms": float(dt.max() * 1000.0),
    }

#python -m rppg.capture 0
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    src = open_source(int(arg) if arg.isdigit() else arg)

    stamps: list[float] = []
    with src:
        for fr in src:
            stamps.append(fr.timestamp)
            cv2.putText(
                fr.image, f"t={fr.timestamp:7.3f}s  #{fr.index}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("capture", fr.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

    print(timing_stats(stamps))
    if isinstance(src, WebcamCapture):
        print(f"dropped frames: {src.dropped_frames}")