# Pipeline rPPG multi-thread: frame -> fata -> ROI -> trace RGB -> resample -> POS -> bandpass -> puls -> netezire.
# Fiecare etapa ruleaza in propriul thread, etapele sunt legate prin cozi.

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .capture import Frame, open_source
from .face import FaceResult, FaceTracker
from .filters import clean_pulse, resample_uniform
from .hr import HrConfig, HrEstimate, HrTracker, TrackerConfig, estimate_hr
from .pos import pos
from .roi import RoiConfig, RoiExtractor, RoiResult
from .signal_trace import SignalTrace, TraceConfig
from .utils import build_dataclass

log = logging.getLogger(__name__)

_INVALID = HrEstimate(float("nan"), float("nan"), float("nan"), False)


# Ruleaza resample, POS, bandpass si spectrul pe un trace. Intoarce estimarea si pulsul.
def estimate_from_trace(t: np.ndarray, rgb: np.ndarray, cfg: "PipelineConfig") -> tuple[HrEstimate, np.ndarray]:
    if len(t) < 2 or t[-1] - t[0] < cfg.hr.min_seconds:
        return _INVALID, np.empty(0)
    _, rgb_u = resample_uniform(t, rgb, cfg.fs)
    pulse = pos(rgb_u, cfg.fs, cfg.pos_window_seconds)
    pulse = clean_pulse(pulse, cfg.fs, cfg.hr.low_hz, cfg.hr.high_hz, cfg.filter_order)
    return estimate_hr(pulse, cfg.fs, cfg.hr), pulse


# Banda filtrului se ia din hr.low_hz si hr.high_hz
@dataclass
class PipelineConfig:
    fs: float = 30.0
    pos_window_seconds: float = 1.6
    filter_order: int = 3
    update_interval: float = 1.0
    face: dict = field(default_factory=dict)
    trace: TraceConfig = field(default_factory=TraceConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    hr: HrConfig = field(default_factory=HrConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    # Construieste config-ul dintr-un dict, sectiunea capture este ignorata
    @classmethod
    def from_dict(cls, data: Optional[dict] = None) -> "PipelineConfig":
        data = dict(data or {})
        data.pop("capture", None)
        return cls(
            trace=build_dataclass(TraceConfig, data.pop("trace", None)),
            roi=build_dataclass(RoiConfig, data.pop("roi", None)),
            hr=build_dataclass(HrConfig, data.pop("hr", None)),
            tracker=build_dataclass(TrackerConfig, data.pop("tracker", None)),
            **data,
        )


# Rezultatul pentru fiecare frame care iese din pipeline.
# is_new = True doar cand s-a calculat o estimare noua (aprox. o data pe update_interval).
@dataclass(frozen=True)
class FrameOutput:
    frame: Frame
    roi: Optional[RoiResult]
    face: Optional[FaceResult]
    bpm: Optional[float]
    estimate: HrEstimate
    pulse: np.ndarray
    duration: float
    is_new: bool
    status: str

    @property
    def timestamp(self) -> float:
        return self.frame.timestamp


# Pune un element in coada, renunta cand pipeline-ul se opreste.
def _put(q: Queue, item, stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except Full:
            pass
    return False


# Ia un element din coada, intoarce None (sfarsit de stream) cand pipeline-ul se opreste.
def _get(q: Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            return q.get(timeout=0.1)
        except Empty:
            pass
    return None


# Foloseste DirectShow pentru webcam pe Windows, backend-ul implicit merge slab acolo.
def default_camera_kwargs(source) -> dict:
    if isinstance(source, int) and sys.platform == "win32":
        return {"backend": cv2.CAP_DSHOW}
    return {}


class RPPGPipeline:
    # source: index camera, cale video sau o sursa deja deschisa (de ex. din build_source)
    # face_tracker: optional, altfel se creeaza FaceTracker in thread-ul de detectie
    def __init__(
        self,
        source=0,
        config: Optional[PipelineConfig] = None,
        camera_kwargs: Optional[dict] = None,
        face_tracker: Optional[FaceTracker] = None,
        queue_size: int = 30,
    ):
        self.cfg = config or PipelineConfig()
        if self.cfg.fs <= 2.0 * self.cfg.hr.high_hz:
            raise ValueError("fs must be above twice hr.high_hz")
        if self.cfg.trace.window_seconds < self.cfg.hr.min_seconds:
            raise ValueError("trace.window_seconds must be at least hr.min_seconds")

        self._source_arg = source
        self._camera_kwargs = default_camera_kwargs(source) if camera_kwargs is None else camera_kwargs

        # Obiectele etapelor sunt expuse pentru statistici si control (skin on/off, rezultate trace)
        self.source = None
        self.face: Optional[FaceTracker] = face_tracker
        self.roi = RoiExtractor(self.cfg.roi)
        self.trace = SignalTrace(self.cfg.trace)
        self.tracker = HrTracker(self.cfg.tracker)

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._reset_gen = 0  # crescut de reset(), etapele isi golesc starea cand il vad schimbat

        self._frame_q: Queue = Queue(maxsize=queue_size)
        self._face_q: Queue = Queue(maxsize=queue_size)
        self._roi_q: Queue = Queue(maxsize=queue_size)
        self._signal_q: Queue = Queue(maxsize=queue_size)
        self._result_q: Queue = Queue(maxsize=queue_size)

    # Porneste thread-urile
    def start(self) -> "RPPGPipeline":
        if self._threads:
            return self
        stages = [
            ("capture", self._capture_stage, self._frame_q),
            ("face", self._face_stage, self._face_q),
            ("roi", self._roi_stage, self._roi_q),
            ("signal", self._signal_stage, self._signal_q),
            ("pulse", self._pulse_stage, self._result_q),
        ]
        for name, fn, out_q in stages:
            t = threading.Thread(target=self._run_stage, args=(name, fn, out_q), name=name, daemon=True)
            self._threads.append(t)
            t.start()
        return self

    # Opreste thread-urile, camera se elibereaza si trackerul de fata se inchide
    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)

    # Goleste starea (trace + tracker), de ex. cand se schimba persoana
    def reset(self) -> None:
        self._reset_gen += 1

    # Intoarce rezultatele frame cu frame, se termina la sfarsitul sursei sau la stop()
    def results(self) -> Iterator[FrameOutput]:
        while not self._stop.is_set():
            try:
                out = self._result_q.get(timeout=0.5)
            except Empty:
                continue
            if out is None:
                return
            yield out

    @property
    def bpm(self) -> Optional[float]:
        return self.tracker.bpm

    def __enter__(self) -> "RPPGPipeline":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # Ruleaza o etapa, logheaza erorile si trimite mereu None mai departe, ca bucla principala sa nu astepte la infinit.
    def _run_stage(self, name: str, fn: Callable[[], None], out_q: Queue) -> None:
        try:
            fn()
        except Exception:
            log.exception("Stage '%s' stopped", name)
        finally:
            _put(out_q, None, self._stop)

    def _capture_stage(self) -> None:
        src = self._source_arg
        if isinstance(src, (int, str)):
            src = open_source(src, **self._camera_kwargs)
        self.source = src
        with src:
            for frame in src:
                if not _put(self._frame_q, frame, self._stop):
                    break

    def _face_stage(self) -> None:
        # Trackerul se creeaza in acest thread (MediaPipe nu iubeste schimbarea de thread)
        tracker = self.face or FaceTracker(**self.cfg.face)
        self.face = tracker
        try:
            while True:
                frame = _get(self._frame_q, self._stop)
                if frame is None:
                    break
                # Frame-urile fara fata trec mai departe, ca afisarea sa nu ingheate.
                if not _put(self._face_q, (frame, tracker.process(frame)), self._stop):
                    break
        finally:
            tracker.close()

    def _roi_stage(self) -> None:
        while True:
            data = _get(self._face_q, self._stop)
            if data is None:
                break
            frame, face = data
            roi = self.roi.extract(frame.image, face) if face is not None else None
            # Frame-urile fara fata sau cu ROI invalid trec mai departe, trace-ul le sare si detecteaza golurile.
            if not _put(self._roi_q, (frame, roi, face), self._stop):
                break

    def _signal_stage(self) -> None:
        next_send = None
        gen = self._reset_gen
        while True:
            data = _get(self._roi_q, self._stop)
            if data is None:
                break
            frame, roi, face = data

            if gen != self._reset_gen:
                gen = self._reset_gen
                self.trace.clear()
                next_send = None

            self.trace.update(frame.timestamp, frame.image, roi)
            snapshot = None
            if next_send is None:
                next_send = frame.timestamp + self.cfg.update_interval
            if frame.timestamp >= next_send:
                next_send = frame.timestamp + self.cfg.update_interval
                snapshot = self.trace.get()

            if not _put(self._signal_q, (frame, roi, face, snapshot, self.trace.duration), self._stop):
                break

    def _pulse_stage(self) -> None:
        estimate = _INVALID
        pulse = np.empty(0)
        gen = self._reset_gen
        while True:
            data = _get(self._signal_q, self._stop)
            if data is None:
                break
            frame, roi, face, snapshot, duration = data

            if gen != self._reset_gen:
                gen = self._reset_gen
                self.tracker.reset()
                estimate = _INVALID
                pulse = np.empty(0)

            is_new = snapshot is not None
            if is_new:
                t, rgb = snapshot
                estimate, p = estimate_from_trace(t, rgb, self.cfg)
                if p.size:
                    pulse = p
                # Trackerul se actualizeaza la fiecare tick, ca sa poata tine sau arunca valoarea cand semnalul se pierde
                self.tracker.update(frame.timestamp, estimate)

            out = FrameOutput(
                frame, roi, face, self.tracker.bpm, estimate, pulse, duration, is_new,
                self._status(face, roi, duration, estimate),
            )
            if not _put(self._result_q, out, self._stop):
                break

    # Text scurt cu starea pipeline-ului, pentru overlay
    def _status(self, face, roi, duration: float, estimate: HrEstimate) -> str:
        if face is None:
            return "no face"
        if roi is None or not roi.valid:
            return "bad roi"
        if duration < self.cfg.hr.min_seconds:
            return f"collecting {duration:.0f}/{self.cfg.hr.min_seconds:.0f}s"
        return "tracking" if estimate.valid else "low quality"