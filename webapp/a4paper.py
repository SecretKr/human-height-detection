import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
A4_ASPECT = A4_HEIGHT_CM / A4_WIDTH_CM


@dataclass
class A4Detection:
    corners: np.ndarray
    width_px: float
    height_px: float
    orientation: str
    cm_per_px: float
    px_per_cm: float


@dataclass
class HeightEstimate:
    height_cm: float
    height_px: float
    used_mask: bool


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _preprocess_a4_mask(bgr: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr, (7, 7), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(l_channel)

    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def detect_a4_paper(
    image: np.ndarray,
    input_color: str = "rgb",
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    min_area_ratio: float = 0.02,
    aspect_tolerance: float = 0.2,
) -> Optional[A4Detection]:
    if input_color.lower() == "bgr":
        bgr = image
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    offset_x = 0
    offset_y = 0
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        x1 = max(0, min(int(x1), bgr.shape[1] - 1))
        y1 = max(0, min(int(y1), bgr.shape[0] - 1))
        x2 = max(0, min(int(x2), bgr.shape[1]))
        y2 = max(0, min(int(y2), bgr.shape[0]))
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            return None
        bgr = bgr[y1:y2, x1:x2]
        offset_x = x1
        offset_y = y1

    mask = _preprocess_a4_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = bgr.shape[:2]
    min_area = min_area_ratio * (h * w)
    best = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype("float32")
        rect = _order_points(pts)

        width_px = float(np.linalg.norm(rect[1] - rect[0]))
        height_px = float(np.linalg.norm(rect[2] - rect[1]))
        if width_px <= 1.0 or height_px <= 1.0:
            continue

        long_px = max(width_px, height_px)
        short_px = min(width_px, height_px)
        ratio = long_px / short_px
        ratio_score = 1.0 - abs(ratio - A4_ASPECT) / A4_ASPECT
        if ratio_score < max(0.0, 1.0 - aspect_tolerance):
            continue

        area_score = area / float(h * w)
        score = ratio_score * area_score
        if score <= best_score:
            continue

        best_score = score
        best = {
            "rect": rect,
            "width_px": width_px,
            "height_px": height_px,
        }

    if best is None:
        return None

    width_px = best["width_px"]
    height_px = best["height_px"]
    long_px = max(width_px, height_px)
    short_px = min(width_px, height_px)
    orientation = "portrait" if height_px >= width_px else "landscape"
    cm_per_px_long = A4_HEIGHT_CM / long_px
    cm_per_px_short = A4_WIDTH_CM / short_px
    cm_per_px = (cm_per_px_long + cm_per_px_short) * 0.5

    corners = best["rect"].copy()
    corners[:, 0] += offset_x
    corners[:, 1] += offset_y

    return A4Detection(
        corners=corners,
        width_px=width_px,
        height_px=height_px,
        orientation=orientation,
        cm_per_px=cm_per_px,
        px_per_cm=1.0 / cm_per_px,
    )


def a4_detection_from_bbox(
    bbox: Tuple[int, int, int, int],
) -> Optional[A4Detection]:
    x1, y1, x2, y2 = bbox
    width_px = float(max(0, x2 - x1))
    height_px = float(max(0, y2 - y1))
    if width_px <= 1.0 or height_px <= 1.0:
        return None

    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype="float32",
    )

    long_px = max(width_px, height_px)
    short_px = min(width_px, height_px)
    orientation = "portrait" if height_px >= width_px else "landscape"

    cm_per_px_long = A4_HEIGHT_CM / long_px
    cm_per_px_short = A4_WIDTH_CM / short_px
    cm_per_px = (cm_per_px_long + cm_per_px_short) * 0.5

    return A4Detection(
        corners=corners,
        width_px=width_px,
        height_px=height_px,
        orientation=orientation,
        cm_per_px=cm_per_px,
        px_per_cm=1.0 / cm_per_px,
    )


def a4_detection_from_mask(mask: np.ndarray) -> Optional[A4Detection]:
    if mask is None:
        return None
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return None

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = _order_points(box.astype("float32"))

    width_px = float(np.linalg.norm(box[1] - box[0]))
    height_px = float(np.linalg.norm(box[2] - box[1]))
    if width_px <= 1.0 or height_px <= 1.0:
        return None

    long_px = max(width_px, height_px)
    short_px = min(width_px, height_px)
    orientation = "portrait" if height_px >= width_px else "landscape"

    cm_per_px_long = A4_HEIGHT_CM / long_px
    cm_per_px_short = A4_WIDTH_CM / short_px
    cm_per_px = (cm_per_px_long + cm_per_px_short) * 0.5

    return A4Detection(
        corners=box,
        width_px=width_px,
        height_px=height_px,
        orientation=orientation,
        cm_per_px=cm_per_px,
        px_per_cm=1.0 / cm_per_px,
    )


def estimate_height_from_bbox(
    bbox: Tuple[int, int, int, int],
    cm_per_px: float,
) -> HeightEstimate:
    x1, y1, x2, y2 = bbox
    height_px = float(max(0, y2 - y1))
    height_cm = height_px * cm_per_px
    return HeightEstimate(height_cm=height_cm, height_px=height_px, used_mask=False)


def estimate_height_from_mask(
    mask: np.ndarray,
    cm_per_px: float,
) -> Optional[HeightEstimate]:
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    height_px = float(ys.max() - ys.min())
    height_cm = height_px * cm_per_px
    return HeightEstimate(height_cm=height_cm, height_px=height_px, used_mask=True)


def estimate_human_height(
    image: np.ndarray,
    a4_detection: A4Detection,
    person_bbox: Optional[Tuple[int, int, int, int]] = None,
    person_mask: Optional[np.ndarray] = None,
) -> Optional[HeightEstimate]:
    if a4_detection is None:
        return None

    if person_mask is not None:
        mask_est = estimate_height_from_mask(person_mask, a4_detection.cm_per_px)
        if mask_est is not None:
            return mask_est

    if person_bbox is None:
        return None

    return estimate_height_from_bbox(person_bbox, a4_detection.cm_per_px)


def draw_a4_overlay(image: np.ndarray, a4_detection: A4Detection) -> np.ndarray:
    overlay = image.copy()
    corners = a4_detection.corners.astype(int)
    cv2.polylines(overlay, [corners], True, (0, 255, 255), 2)

    center = corners.mean(axis=0).astype(int)
    label = f"A4 {a4_detection.orientation} {a4_detection.width_px:.0f}x{a4_detection.height_px:.0f}px"
    cv2.putText(
        overlay,
        label,
        (center[0] - 120, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return overlay


def summarize_detection(a4_detection: Optional[A4Detection]) -> Dict[str, Any]:
    if a4_detection is None:
        return {"detected": False}
    return {
        "detected": True,
        "width_px": round(a4_detection.width_px, 1),
        "height_px": round(a4_detection.height_px, 1),
        "orientation": a4_detection.orientation,
        "cm_per_px": round(a4_detection.cm_per_px, 5),
        "px_per_cm": round(a4_detection.px_per_cm, 3),
    }
