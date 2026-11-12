# Demo live: citeste webcam-ul (sau un fisier video), ruleaza pipeline-ul rPPG multi-thread si afiseaza pulsul pe ecran.
#
#  python scripts/run_webcam.py 0
#  python scripts/run_webcam.py path/to/video.avi
#  python scripts/run_webcam.py 0 --no-skin --mirror --csv outputs/run.csv
#
#  Taste: q = iesire, s = comuta masca de piele, r = reset pipeline

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from rppg.capture import CameraConfig, WebcamCapture, open_source
from rppg.pipeline import PipelineConfig, RPPGPipeline
from rppg.roi import draw_roi
from rppg.utils import FpsMeter, build_dataclass, load_config, setup_logging

GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)


# Citeste optiunile din linia de comanda.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live rPPG heart rate with the POS algorithm (multi-thread)")
    p.add_argument("source", nargs="?", default="0", help="camera index or video file (default 0)")
    p.add_argument("--config", help="YAML config file, same keys as PipelineConfig")
    p.add_argument("--csv", help="write time, bpm, raw bpm and SNR to this CSV file")
    p.add_argument("--mirror", action="store_true", help="flip the preview like a mirror")
    p.add_argument("--no-pulse", action="store_true", help="do not show the pulse waveform window")

    cam = p.add_argument_group("camera")
    cam.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default=None,
                     help="OpenCV backend, auto = DirectShow on Windows (default: from config)")
    cam.add_argument("--unlock-wb", action="store_true", help="do not lock white balance")
    cam.add_argument("--unlock-exposure", action="store_true", help="do not lock exposure")

    roi = p.add_argument_group("roi")
    roi.add_argument("--no-skin", action="store_true", help="turn the skin filter off")
    roi.add_argument("--cr", nargs=2, type=int, metavar=("LOW", "HIGH"), help="skin Cr range")
    roi.add_argument("--cb", nargs=2, type=int, metavar=("LOW", "HIGH"), help="skin Cb range")
    roi.add_argument("--min-skin-fraction", type=float, help="0 = never drop a region")
    return p.parse_args()


# Intoarce constanta de backend OpenCV pentru numele ales, sau None pentru cel implicit.
def pick_backend(name: str):
    if name == "auto":
        return cv2.CAP_DSHOW if sys.platform == "win32" else None
    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[name]


# Deschide webcam-ul pentru un numar sau fisierul video pentru o cale, setarile camerei vin din sectiunea capture.
# Sursa intoarsa este intrata in with de catre thread-ul de captura al pipeline-ului.
def build_source(args: argparse.Namespace, capture: dict):
    if not args.source.isdigit():
        return open_source(args.source)
    kwargs = dict(capture)
    kwargs.pop("index", None)
    backend = args.backend or kwargs.pop("backend", "auto")
    kwargs.pop("backend", None)
    if args.unlock_wb:
        kwargs["lock_white_balance"] = False
    if args.unlock_exposure:
        kwargs["lock_exposure"] = False
    build_dataclass(CameraConfig, kwargs)  # verifica cheile, o greseala in YAML da eroare
    return open_source(int(args.source), backend=pick_backend(backend), **kwargs)


# Construieste config-ul din YAML, apoi aplica optiunile din linia de comanda.
def build_config(args: argparse.Namespace, raw: dict) -> PipelineConfig:
    cfg = PipelineConfig.from_dict(raw)
    if args.no_skin:
        cfg.roi.use_skin_mask = False
    if args.cr:
        cfg.roi.cr_range = tuple(args.cr)
    if args.cb:
        cfg.roi.cb_range = tuple(args.cb)
    if args.min_skin_fraction is not None:
        cfg.roi.min_skin_fraction = args.min_skin_fraction
    return cfg


# Deseneaza semnalul de puls ca grafic, scalat la valoarea maxima.
def draw_pulse(pulse: np.ndarray, width: int = 480, height: int = 120) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    v = np.asarray(pulse, dtype=np.float64)[-width:]
    if len(v) < 2:
        return canvas
    span = np.abs(v).max()
    if span < 1e-9:
        return canvas
    x = np.linspace(0, width - 1, len(v))
    y = height / 2 - v / span * (height / 2 - 4)
    cv2.polylines(canvas, [np.stack([x, y], axis=1).astype(np.int32)], False, GREEN, 1)
    return canvas


# Formateaza un numar pentru CSV, gol cand nu exista valoare.
def fmt(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.2f}"


def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    raw = load_config(args.config) if args.config else {}
    cfg = build_config(args, raw)
    src = build_source(args, raw.get("capture", {}))
    meter = FpsMeter()

    csv_file = writer = None
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["time_s", "bpm", "raw_bpm", "snr_db", "valid"])

    try:
        with RPPGPipeline(source=src, config=cfg) as pipe:
            for out in pipe.results():
                est = out.estimate
                # Un rand in CSV doar cand exista o estimare noua
                if writer is not None and out.is_new:
                    writer.writerow([fmt(out.timestamp), fmt(out.bpm), fmt(est.bpm),
                                     fmt(est.snr_db), int(est.valid)])

                vis = out.frame.image.copy()
                if out.roi is not None:
                    draw_roi(vis, out.roi)
                # Oglindirea se face inainte de text, ca textul sa nu fie oglindit.
                if args.mirror:
                    vis = cv2.flip(vis, 1)

                bpm = f"{out.bpm:.0f} bpm" if out.bpm is not None else "-- bpm"
                ok = out.status == "tracking"
                cv2.putText(vis, f"{bpm}  [{out.status}]", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN if ok else ORANGE, 2)
                snr_text = f"{est.snr_db:.1f} dB" if np.isfinite(est.snr_db) else "--"
                cv2.putText(vis, f"snr {snr_text}  fps {meter.tick():.0f}  skin {'on' if pipe.roi.cfg.use_skin_mask else 'off'}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 1)
                if out.face is None:
                    cv2.putText(vis, "NO FACE", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

                cv2.imshow("rppg", vis)
                if not args.no_pulse:
                    cv2.imshow("pulse", draw_pulse(out.pulse))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    pipe.roi.cfg.use_skin_mask = not pipe.roi.cfg.use_skin_mask
                if key == ord("r"):
                    pipe.reset()

            if pipe.face is not None:
                print(f"face detection rate: {pipe.face.detection_rate:.1%}")
            print(f"trace resets: {pipe.trace.resets}, skipped frames: {pipe.trace.skipped}")
            if isinstance(src, WebcamCapture):
                print(f"dropped frames: {src.dropped_frames}")
    finally:
        cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()