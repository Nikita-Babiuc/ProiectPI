#Turns face landmarks into masks of skin regions (forehead and cheeks) used to extract the RGB signal.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    from .face import FaceResult

log = logging.getLogger(__name__)


FOREHEAD = [67, 109, 10, 338, 297, 334, 296, 336, 9, 107, 66, 105]
RIGHT_CHEEK = [116, 117, 118, 101, 36, 205, 207, 213, 147, 123]
LEFT_CHEEK = [345, 346, 347, 330, 266, 425, 427, 433, 376, 352]

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]

REGIONS: dict[str, list[int]] = {
    "forehead": FOREHEAD,
    "left_cheek": LEFT_CHEEK,
    "right_cheek": RIGHT_CHEEK,
    "face": FACE_OVAL,
}

PRESETS: dict[str, tuple[str, ...]] = {
    "cheeks": ("left_cheek", "right_cheek"),
    "forehead_cheeks": ("forehead", "left_cheek", "right_cheek"),
}


# Expands presets, validates region names and removes duplicates.
def resolve_regions(names: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for n in names:
        for r in PRESETS.get(n, (n,)):
            if r not in REGIONS:
                valid = sorted(list(REGIONS) + list(PRESETS))
                raise ValueError(f"Unknown ROI region '{r}'. Valid: {valid}")
            if r not in out:
                out.append(r)
    return tuple(out)


@dataclass
class RoiConfig:
    regions: Sequence[str] = ("forehead", "cheeks")
    erode_frac: float = 0.02
    use_skin_mask: bool = True
    cr_range: tuple[int, int] = (133, 173)
    cb_range: tuple[int, int] = (77, 127)
    min_luma: int = 30
    max_channel: int = 250
    min_skin_fraction: float = 0.3
    min_pixels: int = 200


@dataclass
class RoiResult:

    mask: np.ndarray
    region_masks: dict[str, np.ndarray]
    n_pixels: int
    valid: bool
    skin_used: bool


# Returns the pixel polygon of a named region.
def region_polygon(landmarks_px: np.ndarray, name: str) -> np.ndarray:
    return np.round(landmarks_px[REGIONS[name]]).astype(np.int32)


# Draws a filled polygon as a binary mask.
def polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


# Returns a mask of skin-colored pixels, excluding dark and saturated ones.
def skin_mask(
    image_bgr: np.ndarray,
    cr_range: tuple[int, int] = (133, 173),
    cb_range: tuple[int, int] = (77, 127),
    min_luma: int = 30,
    max_channel: int = 250,
) -> np.ndarray:
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([min_luma, cr_range[0], cb_range[0]], dtype=np.uint8)
    upper = np.array([255, cr_range[1], cb_range[1]], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask[image_bgr.max(axis=2) >= max_channel] = 0
    return mask


class RoiExtractor:
    # Stores the config and resolves the region names.
    def __init__(self, config: Optional[RoiConfig] = None):
        self.cfg = config or RoiConfig()
        self.regions = resolve_regions(self.cfg.regions)

    # Builds the ROI mask for one frame from the face landmarks, with erosion and skin filtering.
    def extract(self, image_bgr: np.ndarray, face: "FaceResult") -> RoiResult:
        h, w = image_bgr.shape[:2]
        cfg = self.cfg

        face_w = max(face.bbox[2] - face.bbox[0], 1)
        k = int(round(cfg.erode_frac * face_w))
        kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            if k > 0
            else None
        )
        masks: dict[str, np.ndarray] = {}
        for name in self.regions:
            m = polygon_mask((h, w), region_polygon(face.landmarks_px, name))
            if kernel is not None:
                m = cv2.erode(m, kernel)
            masks[name] = m

        skin_used = False
        if cfg.use_skin_mask:
            skin = skin_mask(
                image_bgr, cfg.cr_range, cfg.cb_range, cfg.min_luma, cfg.max_channel
            )
            kept: dict[str, np.ndarray] = {}
            for name, m in masks.items():
                n_geo = cv2.countNonZero(m)
                if n_geo == 0:
                    continue
                sm = cv2.bitwise_and(m, skin)
                if cv2.countNonZero(sm) >= cfg.min_skin_fraction * n_geo:
                    kept[name] = sm
                else:
                    log.debug("ROI '%s' dropped: too few skin pixels", name)
            if kept:
                masks, skin_used = kept, True
            else:
                log.debug("Skin filter rejected all regions; using geometric ROI")

        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks.values():
            cv2.bitwise_or(combined, m, dst=combined)
        n = cv2.countNonZero(combined)

        return RoiResult(
            mask=combined,
            region_masks=masks,
            n_pixels=n,
            valid=n >= cfg.min_pixels,
            skin_used=skin_used,
        )


_COLORS = {
    "forehead": (0, 0, 255),
    "left_cheek": (255, 0, 0),
    "right_cheek": (0, 200, 0),
    "face": (0, 200, 255),
}


# Tints the ROI regions on the image
def draw_roi(image: np.ndarray, roi: RoiResult, alpha: float = 0.4) -> np.ndarray:
    overlay = image.copy()
    for name, m in roi.region_masks.items():
        overlay[m > 0] = _COLORS.get(name, (255, 255, 0))
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)
    return image


if __name__ == "__main__":
    import sys

    from .capture import open_source
    from .face import FaceTracker

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    src = open_source(int(arg) if arg.isdigit() else arg)
    extractor = RoiExtractor(RoiConfig(regions=("forehead", "cheeks")))

    with src, FaceTracker() as tracker:
        for frame in src:
            vis = frame.image.copy()
            face = tracker.process(frame)
            if face is None:
                text, color = "NO FACE", (0, 0, 255)
            else:
                roi = extractor.extract(frame.image, face)
                draw_roi(vis, roi)
                b, g, r, _ = cv2.mean(frame.image, mask=roi.mask)
                text = (
                    f"px={roi.n_pixels} valid={roi.valid} skin={roi.skin_used} "
                    f"RGB=({r:.0f},{g:.0f},{b:.0f})"
                )
            color = (0, 255, 0) if roi.valid else (0, 165, 255)
            cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imshow("roi", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                extractor.cfg.use_skin_mask = not extractor.cfg.use_skin_mask
    cv2.destroyAllWindows()