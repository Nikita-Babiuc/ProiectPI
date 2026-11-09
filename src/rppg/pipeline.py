import threading
from queue import Queue


from capture import open_source
from face import FaceTracker
from roi import RoiExtractor



def start_pipeline():
    frame_queue = Queue()
    face_queue = Queue()
    roi_queue = Queue()

    t1 = threading.Thread(
        target=capture_stage,
        args=(frame_queue,)
    )

    t2 = threading.Thread(
        target = face_detection_stage(),
        args=(face_queue,)
    )
    t3 = threading.Thread(
        target = roi_stage,
        args=(roi_queue,)
    )

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

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

