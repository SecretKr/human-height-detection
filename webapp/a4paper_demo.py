import argparse
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

from a4paper import detect_a4_paper, draw_a4_overlay, estimate_height_from_bbox


def parse_bbox(text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be x1,y1,x2,y2")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description="A4 detection demo")
    parser.add_argument("--image", type=str, default=None, help="Path to an RGB image")
    parser.add_argument("--bbox", type=str, default=None, help="Person bbox x1,y1,x2,y2")
    parser.add_argument("--bgr", action="store_true", help="Treat input as BGR instead of RGB")
    args = parser.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print("Failed to read image", file=sys.stderr)
            return 1
        input_color = "bgr" if args.bgr else "rgb"
        if not args.bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam", file=sys.stderr)
            return 1
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture frame", file=sys.stderr)
            return 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_color = "rgb"

    a4 = detect_a4_paper(image, input_color=input_color)
    if a4 is None:
        print("A4 not detected")
        return 2

    if args.bbox:
        bbox = parse_bbox(args.bbox)
        est = estimate_height_from_bbox(bbox, a4.cm_per_px)
        print(f"Estimated height: {est.height_cm:.1f} cm (px: {est.height_px:.1f})")

    display = draw_a4_overlay(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), a4)
    cv2.imshow("A4 detection", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
