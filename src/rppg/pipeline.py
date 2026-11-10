import threading
from queue import Queue, Empty
from tabnanny import NannyNag

import cv2

from .capture import open_source
from .face import FaceTracker
from .filters import clean_pulse, resample_uniform
from .hr import HrEstimate, HrTracker, estimate_hr
from .pos import pos
from .roi import RoiExtractor, draw_roi
from .signal_trace import SignalTrace




def start_pipeline():
    frame_queue = Queue(maxsize=30)
    face_queue = Queue(maxsize=30)
    roi_queue = Queue(maxsize=30)
    signal_queue  = Queue(maxsize=30)
    result_queue = Queue(maxsize=30)

    t1 = threading.Thread(
        target=capture_stage,
        args=(frame_queue,),
        daemon=True
    )

    t2 = threading.Thread(
        target = face_detection_stage,
        args=(frame_queue,face_queue),
        daemon=True
    )
    t3 = threading.Thread(
        target = roi_stage,
        args=(face_queue,roi_queue,),
        daemon=True
    )

    t4 = threading.Thread(
        target = signal_stage,
        args=(roi_queue,signal_queue,),
        daemon=True
    )
    t5 = threading.Thread(
        target = pulse_stage,
        args=(signal_queue,result_queue,),
        daemon=True
    )

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    try:
        while True:
            try:
                result = result_queue.get(timeout=0.5)
            except Empty:
                continue

            if result is None:
                break

            frame, roi, face, bpm, estimate, duration, is_new = result

            if is_new:
                shown = "--" if bpm is None else f"{bpm:.1f}"
                print(
                    f"t={frame.timestamp:6.1f}s  bpm={shown:>5}  raw={estimate.bpm:6.1f}  "
                    f"snr={estimate.snr_db:5.1f} dB  valid={estimate.valid}"
                )

            cv2.imshow("heart rate", draw_frame(frame, roi, face, bpm, estimate, duration))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("stopped")
                return

        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        cv2.destroyAllWindows()


def capture_stage(frame_queue):

    source = open_source(0)

    with source:
        for frame in source:
            frame_queue.put(frame)

    frame_queue.put(None)

def face_detection_stage(frame_queue, face_queue):

    with FaceTracker() as tracker:

        while True:

            frame = frame_queue.get()

            if frame is None:
                break

            face = tracker.process(frame)

            if face is not None:
                face_queue.put((frame, face))

        face_queue.put(None)

def roi_stage(face_queue, roi_queue):

    extractor = RoiExtractor()

    while True:
        data = face_queue.get()

        if data is None:
            break

        frame, face = data

        roi = extractor.extract(frame.image, face)

        if roi.valid:
            roi_queue.put((frame, roi, face))

    roi_queue.put(None)

def signal_stage(roi_queue, signal_queue):
    trace = SignalTrace()

    last_sent = 0.0

    while True:
        data = roi_queue.get()

        if data is None:
            break

        frame, roi, face = data

        trace.update(frame.timestamp, frame.image, roi)
        snapshot = None
        if frame.timestamp - last_sent >= 1.0:
            last_sent = frame.timestamp
            snapshot = trace.get()

        signal_queue.put((frame, roi, face, snapshot, trace.duration))

    signal_queue.put(None)


def pulse_stage(signal_queue, result_queue):
    tracker = HrTracker()
    estimate = None

    while True:
        data = signal_queue.get()

        if data is None:
            break

        frame, roi, face, snapshot, duration = data

        is_new = snapshot is not None

        if is_new:
            t, rgb = snapshot

            if len(t) >= 2:
                _, rgb_uniform = resample_uniform(t, rgb, 30)
                pulse = clean_pulse(pos(rgb_uniform, 30), 30)
                estimate = estimate_hr(pulse, 30)
            else:
                estimate = HrEstimate(float("nan"), float("nan"), float("nan"), False)
            tracker.update(frame.timestamp, estimate)

        result_queue.put((frame, roi, face, tracker.bpm, estimate, duration, is_new))

    result_queue.put(None)


def draw_frame(frame, roi, face, bpm, estimate, duration):
    image = frame.image.copy()

    if roi is not None:
        draw_roi(image, roi)

    if face is None:
        status, color = "no face", (0, 0, 255)
    elif estimate is None or duration < 5:
        status, color = f"collecting {duration:.0f}/5 s", (0, 165, 255)
    else:
        status = f"raw {estimate.bpm:.1f}  snr {estimate.snr_db:.1f} dB  valid {estimate.valid}"
        color = (0, 255, 0) if estimate.valid else (0, 0, 255)

    height = image.shape[0]
    title = "HR --" if bpm is None else f"HR {bpm:.0f} bpm"
    cv2.putText(image, title, (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(image, status, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return image
if __name__ == "__main__":
    start_pipeline()
