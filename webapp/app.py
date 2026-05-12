import base64
import math
import os

import cv2
import cv2.aruco as aruco
import numpy as np
import torch
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO, SAM

A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
A4_ASPECT = A4_HEIGHT_CM / A4_WIDTH_CM

app = Flask(__name__)
app.config["SECRET_KEY"] = "height-detection-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# --- Global State ---
yolo_model = None
sam_model = None
a4_seg_model = None
device = None
cap = None
streaming = False
detection_mode = "aruco"

# ArUco setup
MARKER_SIZE = 6  # cm
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
param_markers = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(marker_dict, param_markers)

# Detection state
CONF_THRESHOLD = 0.6
EDGE_MARGIN = 15
frame_count = 0
skip_rate = 3
person_bbox = None
is_cut_off = False
warning_message = ""
distance = None
focal_length = None
marker_corners = None
last_a4_detection = None
all_persons = []
selected_person_idx = -1
a4_seg_class = None
a4_seg_conf = 0.25
a4_seg_model_path = None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def load_models():
    global yolo_model, sam_model, a4_seg_model, device, a4_seg_class, a4_seg_conf, a4_seg_model_path
    device = get_device()
    print(f"Using device: {device}")
    print("Loading YOLO and SAM models...")
    yolo_model = YOLO("yolo26n.pt")
    sam_model = SAM("sam_b.pt")
    a4_seg_model_path = _resolve_a4_seg_model_path()
    a4_seg_class = _resolve_a4_seg_class()
    a4_seg_conf = _resolve_a4_seg_conf()
    if a4_seg_model_path:
        try:
            print(f"Loading A4 segmentation model: {a4_seg_model_path}")
            a4_seg_model = YOLO(a4_seg_model_path)
        except Exception as exc:
            print(f"Failed to load A4 segmentation model: {exc}")
            a4_seg_model = None
    print("Models loaded successfully.")


def _resolve_a4_seg_model_path():
    env_path = os.getenv("A4_SEG_MODEL")
    if env_path:
        return env_path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(repo_root, "model", "a4paper.pt")
    if os.path.exists(model_path):
        return model_path
    return None


def _resolve_a4_seg_class():
    value = os.getenv("A4_SEG_CLASS")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _resolve_a4_seg_conf():
    value = os.getenv("A4_SEG_CONF")
    if value is None or value == "":
        return 0.25
    try:
        return float(value)
    except ValueError:
        return 0.25


def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode("utf-8")


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _a4_detection_from_mask(mask: np.ndarray):
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

    return {
        "corners": box,
        "width_px": width_px,
        "height_px": height_px,
        "orientation": orientation,
        "cm_per_px": cm_per_px,
        "px_per_cm": 1.0 / cm_per_px,
    }


def _a4_detection_from_bbox(bbox):
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

    return {
        "corners": corners,
        "width_px": width_px,
        "height_px": height_px,
        "orientation": orientation,
        "cm_per_px": cm_per_px,
        "px_per_cm": 1.0 / cm_per_px,
    }


def _detect_a4_paper_yolo(
    model,
    image: np.ndarray,
    input_color: str = "rgb",
    roi_bbox=None,
    class_id=None,
    conf: float = 0.25,
    aspect_tolerance: float = 0.2,
    device_override: str | None = None,
    imgsz: int = 640,
):
    if model is None:
        return None

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

    try:
        results = model.predict(
            source=bgr,
            imgsz=imgsz,
            device=device_override,
            conf=conf,
            verbose=False,
        )
    except Exception:
        return None

    if not results:
        return None

    result = results[0]
    if result.masks is None and result.boxes is None and getattr(result, "obb", None) is None:
        return None

    masks = result.masks.data if result.masks is not None else None

    best = None
    best_ratio_diff = None
    best_area = 0.0

    if masks is not None and len(masks) > 0:
        for i, mask_tensor in enumerate(masks):
            if class_id is not None and result.boxes is not None:
                try:
                    if int(result.boxes.cls[i].item()) != int(class_id):
                        continue
                except Exception:
                    continue

            mask = mask_tensor.detach().cpu().numpy()
            if mask.shape[0] != bgr.shape[0] or mask.shape[1] != bgr.shape[1]:
                mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            detection = _a4_detection_from_mask(mask)
            if detection is None:
                continue

            long_px = max(detection["width_px"], detection["height_px"])
            short_px = min(detection["width_px"], detection["height_px"])
            ratio = long_px / max(short_px, 1e-6)
            ratio_diff = abs(ratio - A4_ASPECT)
            if ratio_diff > A4_ASPECT * aspect_tolerance:
                continue

            area = float(np.count_nonzero(mask))
            if best is not None:
                if ratio_diff > best_ratio_diff:
                    continue
                if ratio_diff == best_ratio_diff and area <= best_area:
                    continue

            best_ratio_diff = ratio_diff
            best_area = area
            best = detection

    if best is None and result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            if class_id is not None:
                try:
                    if int(box.cls.item()) != int(class_id):
                        continue
                except Exception:
                    continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detection = _a4_detection_from_bbox((x1, y1, x2, y2))
            if detection is None:
                continue

            long_px = max(detection["width_px"], detection["height_px"])
            short_px = min(detection["width_px"], detection["height_px"])
            ratio = long_px / max(short_px, 1e-6)
            ratio_diff = abs(ratio - A4_ASPECT)
            if ratio_diff > A4_ASPECT * aspect_tolerance:
                continue

            area = detection["width_px"] * detection["height_px"]
            if best is not None:
                if ratio_diff > best_ratio_diff:
                    continue
                if ratio_diff == best_ratio_diff and area <= best_area:
                    continue

            best_ratio_diff = ratio_diff
            best_area = area
            best = detection

    if best is None and getattr(result, "obb", None) is not None and len(result.obb) > 0:
        for i, obb in enumerate(result.obb):
            if class_id is not None:
                try:
                    if int(obb.cls.item()) != int(class_id):
                        continue
                except Exception:
                    continue
            try:
                x1, y1, x2, y2 = map(int, obb.xyxy[0].tolist())
            except Exception:
                continue

            detection = _a4_detection_from_bbox((x1, y1, x2, y2))
            if detection is None:
                continue

            long_px = max(detection["width_px"], detection["height_px"])
            short_px = min(detection["width_px"], detection["height_px"])
            ratio = long_px / max(short_px, 1e-6)
            ratio_diff = abs(ratio - A4_ASPECT)
            if ratio_diff > A4_ASPECT * aspect_tolerance:
                continue

            area = detection["width_px"] * detection["height_px"]
            if best is not None:
                if ratio_diff > best_ratio_diff:
                    continue
                if ratio_diff == best_ratio_diff and area <= best_area:
                    continue

            best_ratio_diff = ratio_diff
            best_area = area
            best = detection

    if best is None:
        return None

    if offset_x or offset_y:
        best["corners"] = best["corners"].copy()
        best["corners"][:, 0] += offset_x
        best["corners"][:, 1] += offset_y

    return best


def _estimate_height_from_bbox(bbox, cm_per_px):
    x1, y1, x2, y2 = bbox
    height_px = float(max(0, y2 - y1))
    height_cm = height_px * cm_per_px
    return {"height_cm": height_cm, "height_px": height_px, "used_mask": False}


def _estimate_height_from_mask(mask: np.ndarray, cm_per_px: float):
    if mask is None:
        return None
    ys, _ = np.where(mask > 0)
    if ys.size == 0:
        return None
    height_px = float(ys.max() - ys.min())
    height_cm = height_px * cm_per_px
    return {"height_cm": height_cm, "height_px": height_px, "used_mask": True}


def _estimate_human_height(cm_per_px, person_bbox=None, person_mask=None):
    if person_mask is not None:
        mask_est = _estimate_height_from_mask(person_mask, cm_per_px)
        if mask_est is not None:
            return mask_est
    if person_bbox is None:
        return None
    return _estimate_height_from_bbox(person_bbox, cm_per_px)


def _draw_a4_overlay(image: np.ndarray, a4_detection) -> np.ndarray:
    overlay = image.copy()
    corners = a4_detection["corners"].astype(int)
    cv2.polylines(overlay, [corners], True, (0, 255, 255), 2)

    center = corners.mean(axis=0).astype(int)
    label = (
        f"A4 {a4_detection['orientation']} "
        f"{a4_detection['width_px']:.0f}x{a4_detection['height_px']:.0f}px"
    )
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/aruco")
def aruco_page():
    return render_template("aruco.html")


@app.route("/api/generate-aruco")
def generate_aruco():
    marker_id = request.args.get("id", 0, type=int)
    size_px = request.args.get("size", 200, type=int)

    # Clamp values to safe ranges
    marker_id = max(0, min(marker_id, 49))
    size_px = max(50, min(size_px, 1000))

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    marker_image = aruco.generateImageMarker(dictionary, marker_id, size_px)

    _, buffer = cv2.imencode(".png", marker_image)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return jsonify({"image": encoded, "id": marker_id, "size": size_px})


@socketio.on("start_stream")
def handle_start_stream(data=None):
    global cap, streaming, frame_count, person_bbox, is_cut_off, warning_message, distance, focal_length, marker_corners, all_persons, selected_person_idx, detection_mode, last_a4_detection

    if streaming:
        return

    mode = "aruco"
    if isinstance(data, dict):
        mode = data.get("mode", "aruco")
    if mode not in ("aruco", "a4"):
        mode = "aruco"
    detection_mode = mode

    # Try DirectShow backend first (Windows), then default
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        emit("error", {"message": "Could not open webcam. Make sure a camera is connected and not in use by another app."})
        return

    streaming = True
    frame_count = 0
    person_bbox = None
    is_cut_off = False
    warning_message = ""
    distance = None
    focal_length = None
    marker_corners = None
    last_a4_detection = None
    all_persons = []
    selected_person_idx = -1

    emit("stream_started")
    print("Stream started")

    # Run the streaming loop in a background thread so the handler returns
    socketio.start_background_task(stream_loop)


def stream_loop():
    global cap, streaming, frame_count, person_bbox, is_cut_off, distance, focal_length, marker_corners, warning_message, all_persons, selected_person_idx, detection_mode, last_a4_detection, a4_seg_model, a4_seg_class, a4_seg_conf

    while streaming and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        display_frame = frame.copy()
        frame_count += 1

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # YOLO + calibration target detection on every nth frame
        if frame_count % skip_rate == 0:
            results = yolo_model(frame, classes=[0], device=device, imgsz=320, conf=CONF_THRESHOLD, verbose=False)

            if len(results[0].boxes) > 0:
                frame_center_x = w / 2
                frame_center_y = h / 2
                min_dist = float('inf')
                
                # Clear previous frame's list
                all_persons = []
                
                # Find the person closest to the middle
                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    all_persons.append([x1, y1, x2, y2])
                    
                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2
                    
                    # Calculate Euclidean distance to frame center
                    dist = math.hypot(box_center_x - frame_center_x, box_center_y - frame_center_y)
                    
                    if dist < min_dist:
                        min_dist = dist
                        person_bbox = [x1, y1, x2, y2]
                        selected_person_idx = i

                # Edge check ONLY for the selected person
                x1, y1, x2, y2 = person_bbox
                if y1 <= EDGE_MARGIN:
                    is_cut_off = True
                    warning_message = "Head is cut off!"
                elif y2 >= h - EDGE_MARGIN:
                    is_cut_off = True
                    warning_message = "Feet are cut off!"
                else:
                    is_cut_off = False
                    warning_message = ""
            else:
                person_bbox = None
                all_persons = []
                selected_person_idx = -1
                is_cut_off = False
                warning_message = ""

            if detection_mode == "aruco":
                # Detect ArUco markers
                detected_corners, marker_IDs, _ = aruco_detector.detectMarkers(gray_frame)
                marker_corners = detected_corners
                last_a4_detection = None

                if marker_IDs is not None:
                    for ids, corners in zip(marker_IDs, detected_corners):
                        fl = 0.9 * frame.shape[1]
                        center = (frame.shape[1] / 2, frame.shape[0] / 2)
                        cam_mat = np.array(
                            [[fl, 0, center[0]],
                             [0, fl, center[1]],
                             [0, 0, 1]], dtype="double"
                        )
                        dist_coef = np.zeros((4, 1))
                        half = MARKER_SIZE / 2.0
                        obj_points = np.array([
                            [-half,  half, 0],
                            [ half,  half, 0],
                            [ half, -half, 0],
                            [-half, -half, 0],
                        ], dtype=np.float32)
                        _, rvec, tvec = cv2.solvePnP(obj_points, corners.reshape(4, 2), cam_mat, dist_coef)
                        tvec = tvec.flatten()
                        distance = math.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
                        focal_length = fl
                else:
                    distance = None
                    focal_length = None
            else:
                marker_corners = None
                distance = None
                focal_length = None
                if a4_seg_model is not None:
                    last_a4_detection = _detect_a4_paper_yolo(
                        a4_seg_model,
                        frame,
                        input_color="bgr",
                        class_id=a4_seg_class,
                        conf=a4_seg_conf,
                        device_override=device,
                    )
                else:
                    last_a4_detection = None

        # Draw all bounding boxes
        for i, bbox in enumerate(all_persons):
            x1, y1, x2, y2 = bbox
            if i == selected_person_idx:
                # The selected (centered) person
                color = (0, 0, 255) if is_cut_off else (0, 255, 0) # Red if cut off, Green if good
            else:
                # Other people Yellow
                color = (0, 255, 255)
                
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

        # Draw calibration overlay
        if detection_mode == "aruco" and marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(display_frame, marker_corners)
        if detection_mode == "a4" and last_a4_detection is not None:
            display_frame = _draw_a4_overlay(display_frame, last_a4_detection)

        # Build status info
        status = {
            "detection_mode": detection_mode,
            "person_detected": person_bbox is not None,
            "is_cut_off": is_cut_off,
            "warning": warning_message,
            "aruco_detected": distance is not None if detection_mode == "aruco" else False,
            "distance": round(distance, 1) if distance and detection_mode == "aruco" else None,
            "a4_detected": last_a4_detection is not None if detection_mode == "a4" else False,
            "a4_cm_per_px": round(last_a4_detection["cm_per_px"], 5) if last_a4_detection and detection_mode == "a4" else None,
        }

        frame_data = encode_frame(display_frame)
        socketio.emit("frame", {"image": frame_data, "status": status})
        socketio.sleep(0.03)

    if cap:
        cap.release()
    streaming = False
    print("Stream stopped")


@socketio.on("stop_stream")
def handle_stop_stream():
    global streaming, cap
    streaming = False
    if cap:
        cap.release()
        cap = None
    emit("stream_stopped")
    print("Stream stopped by client")


@socketio.on("capture")
def handle_capture():
    global cap, streaming, person_bbox, is_cut_off, distance, focal_length, detection_mode

    if not streaming or cap is None or not cap.isOpened():
        emit("capture_result", {"success": False, "error": "Stream not active"})
        return

    if person_bbox is None:
        emit("capture_result", {"success": False, "error": "No person detected"})
        return

    if is_cut_off:
        emit("capture_result", {"success": False, "error": f"Cannot capture: {warning_message}"})
        return

    if detection_mode == "aruco":
        if distance is None or focal_length is None:
            emit("capture_result", {"success": False, "error": "No ArUco marker detected for distance calibration"})
            return

    # Grab the current frame
    ret, frame = cap.read()
    if not ret:
        emit("capture_result", {"success": False, "error": "Failed to capture frame"})
        return

    h, w = frame.shape[:2]

    # Run SAM segmentation
    pad = 10
    px1, py1 = max(0, person_bbox[0] - pad), max(0, person_bbox[1] - pad)
    px2, py2 = min(w, person_bbox[2] + pad), min(h, person_bbox[3] + pad)
    padded_bbox = [[px1, py1, px2, py2]]

    try:
        sam_results = sam_model(frame, bboxes=padded_bbox, device=device, verbose=False)
    except Exception as e:
        emit("capture_result", {"success": False, "error": f"SAM error: {str(e)}"})
        return

    result_frame = frame.copy()
    mask = None
    polygon = None

    if sam_results[0].masks is not None and len(sam_results[0].masks.xy) > 0:
        polygon = sam_results[0].masks.xy[0]
        mask = sam_results[0].masks.data[0].cpu().numpy()

    if detection_mode == "aruco":
        if polygon is None:
            emit("capture_result", {"success": False, "error": "SAM failed to generate mask"})
            return

        highest_y = int(np.min(polygon[:, 1]))
        lowest_y = int(np.max(polygon[:, 1]))
        center_x = int(np.mean(polygon[:, 0]))

        pixel_height = lowest_y - highest_y
        estimated_height = (pixel_height * distance) / focal_length

        # Draw on result frame
        cv2.circle(result_frame, (center_x, highest_y), 5, (0, 0, 255), -1)
        cv2.circle(result_frame, (center_x, lowest_y), 5, (0, 0, 255), -1)
        cv2.line(result_frame, (center_x, highest_y), (center_x, lowest_y), (255, 0, 0), 2)

        # Draw mask overlay
        if mask is not None:
            mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))
            overlay = result_frame.copy()
            overlay[mask_resized > 0] = overlay[mask_resized > 0] * 0.6 + np.array([0, 120, 255]) * 0.4
            result_frame = overlay.astype(np.uint8)

            # Redraw measurement lines on top of overlay
            cv2.circle(result_frame, (center_x, highest_y), 7, (0, 0, 255), -1)
            cv2.circle(result_frame, (center_x, lowest_y), 7, (0, 0, 255), -1)
            cv2.line(result_frame, (center_x, highest_y), (center_x, lowest_y), (255, 255, 0), 2)

        frame_data = encode_frame(result_frame)
        emit("capture_result", {
            "success": True,
            "image": frame_data,
            "height_cm": round(estimated_height, 1),
            "pixel_height": pixel_height,
            "distance_cm": round(distance, 1),
            "detection_mode": "aruco",
        })
        return

    if a4_seg_model is None:
        emit("capture_result", {"success": False, "error": "A4 model not loaded"})
        return

    a4_detection = _detect_a4_paper_yolo(
        a4_seg_model,
        frame,
        input_color="bgr",
        class_id=a4_seg_class,
        conf=a4_seg_conf,
        device_override=device,
    )
    if a4_detection is None:
        emit("capture_result", {"success": False, "error": "No A4 paper detected for scale calibration"})
        return

    height_est = _estimate_human_height(
        a4_detection["cm_per_px"],
        person_bbox=tuple(person_bbox),
        person_mask=mask,
    )

    if height_est is None:
        emit("capture_result", {"success": False, "error": "Failed to estimate height from A4 scale"})
        return

    result_frame = _draw_a4_overlay(result_frame, a4_detection)

    if polygon is not None:
        highest_y = int(np.min(polygon[:, 1]))
        lowest_y = int(np.max(polygon[:, 1]))
        center_x = int(np.mean(polygon[:, 0]))
    else:
        x1, y1, x2, y2 = person_bbox
        highest_y, lowest_y = y1, y2
        center_x = int((x1 + x2) / 2)

    pixel_height = height_est["height_px"]

    cv2.circle(result_frame, (center_x, highest_y), 5, (0, 0, 255), -1)
    cv2.circle(result_frame, (center_x, lowest_y), 5, (0, 0, 255), -1)
    cv2.line(result_frame, (center_x, highest_y), (center_x, lowest_y), (255, 0, 0), 2)

    if mask is not None:
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))
        overlay = result_frame.copy()
        overlay[mask_resized > 0] = overlay[mask_resized > 0] * 0.6 + np.array([0, 120, 255]) * 0.4
        result_frame = overlay.astype(np.uint8)

    frame_data = encode_frame(result_frame)
    emit("capture_result", {
        "success": True,
        "image": frame_data,
        "height_cm": round(height_est["height_cm"], 1),
        "pixel_height": round(pixel_height, 1),
        "cm_per_px": round(a4_detection["cm_per_px"], 5),
        "detection_mode": "a4",
    })


@socketio.on("disconnect")
def handle_disconnect():
    global streaming, cap
    streaming = False
    if cap:
        cap.release()
        cap = None
    print("Client disconnected")


if __name__ == "__main__":
    load_models()
    print("\nStarting web app at http://localhost:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)
