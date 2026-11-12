import sys
from pathlib import Path

import cv2

# Lets the script run without installing the package or setting PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rppg.capture import open_source
from rppg.face import FaceTracker
from rppg.roi import RoiConfig, RoiExtractor, draw_roi, skin_mask

arg = sys.argv[1] if len(sys.argv) > 1 else "0"
if arg.isdigit():
    # DirectShow only exists on Windows.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else None
    src = open_source(int(arg), backend=backend, lock_white_balance=False, lock_exposure=False)
else:
    src = open_source(arg)
extractor = RoiExtractor(RoiConfig(use_skin_mask=False))  # geometric regions only

with src, FaceTracker() as tracker:
    for frame in src:
        vis = frame.image.copy()
        face = tracker.process(frame)
        if face is None:
            cv2.putText(vis, "NO FACE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            roi = extractor.extract(frame.image, face)
            skin = skin_mask(frame.image)
            ycrcb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2YCrCb)
            draw_roi(vis, roi)
            for i, (name, m) in enumerate(roi.region_masks.items()):
                frac = cv2.countNonZero(cv2.bitwise_and(m, skin)) / max(cv2.countNonZero(m), 1)
                color = (0, 255, 0) if frac >= 0.3 else (0, 0, 255)
                cv2.putText(vis, f"{name}: {frac:.0%} skin", (10, 25 + 25 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y, cr, cb, _ = cv2.mean(ycrcb, mask=m)
                cv2.putText(vis, f"{name}: Y={y:.0f} Cr={cr:.0f} Cb={cb:.0f}", (10, 110 + 25 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("debug roi", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cv2.destroyAllWindows()