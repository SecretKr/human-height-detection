# Webapp helpers

## A4 paper detection utility

This folder contains a small helper module to detect an A4 sheet in an RGB frame and estimate a person height from pixel scale.

### Quick demo

Run on a still image:

```bash
python a4paper_demo.py --image path\to\image.jpg
```

Estimate height with a known bounding box:

```bash
python a4paper_demo.py --image path\to\image.jpg --bbox 100,40,240,560
```

Use the webcam (first frame):

```bash
python a4paper_demo.py
```

Notes:
- Input images are assumed RGB; use `--bgr` if you pass a BGR image.
- A4 is 21.0 cm x 29.7 cm; detection validates the aspect ratio.

## A4 segmentation YOLO (optional)

If you have a pretrained segmentation model that can segment A4 sheets, place the weights next to `app.py` and set:

```bash
set A4_SEG_MODEL=yolo11n-seg.pt
```

If the model uses a class ID that is not "paper/document/page", set:

```bash
set A4_SEG_CLASS=0
```

When the file is present, A4 mode uses the segmenter; otherwise it falls back to contour detection.
