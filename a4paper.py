import cv2

from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import os

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check permissions or camera index.")

print("Starting webcam inference. Press 'q' in the preview window to exit.")
use_gui = True
frame_count = 0
max_frames = None  # Set to an int to auto-stop in notebook mode.
model = YOLO(os.path.join("model", "a4paper.pt"))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    results = model.predict(
        source=frame,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        conf=0.25,
        verbose=False,
    )

    annotated = results[0].plot()

    if use_gui:
        try:
            cv2.imshow("YOLOv8 OBB Webcam", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        except cv2.error:
            use_gui = False
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    clear_output(wait=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb)
    plt.axis("off")
    display(plt.gcf())
    plt.close()

    frame_count += 1
    if max_frames is not None and frame_count >= max_frames:
        break

cap.release()
try:
    cv2.destroyAllWindows()
except cv2.error:
    pass