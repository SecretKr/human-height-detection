import base64
import math

import cv2
import cv2.aruco as aruco
import numpy as np
import torch
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO, SAM

from a4paper import (
    A4_ASPECT,
    A4_HEIGHT_CM,
    A4_WIDTH_CM,
    detect_a4_paper,
    draw_a4_overlay,
    estimate_human_height,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "height-detection-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# --- Global State ---
yolo_model = None
sam_model = None
device = None
cap = None
streaming = False
detection_mode = "aruco"
debug_cap = None
debug_streaming = False
debug_last_person_bbox = None

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

A4_DEBUG_DEFAULTS = {
    "blur_ksize": 7,
    "clahe_clip": 3.0,
    "clahe_grid": 8,
    "adapt_block": 21,
    "adapt_c": 10,
    "morph_ksize": 5,
    "close_iter": 2,
    "open_iter": 1,
    "min_area_ratio": 0.02,
    "aspect_tolerance": 0.2,
}

a4_debug_params = A4_DEBUG_DEFAULTS.copy()


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
    global yolo_model, sam_model, device
    device = get_device()
    print(f"Using device: {device}")
    print("Loading YOLO and SAM models...")
    yolo_model = YOLO("yolo26n.pt")
    sam_model = SAM("sam_b.pt")
    print("Models loaded successfully.")


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


def _sanitize_a4_debug_params(data):
    params = A4_DEBUG_DEFAULTS.copy()
    if isinstance(data, dict):
        params.update(data)

    def _clamp_int(value, low, high, default):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return default
        return max(low, min(high, value))

    def _clamp_float(value, low, high, default):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return max(low, min(high, value))

    params["blur_ksize"] = _clamp_int(params["blur_ksize"], 1, 31, 7)
    if params["blur_ksize"] % 2 == 0:
        params["blur_ksize"] += 1

    params["clahe_clip"] = _clamp_float(params["clahe_clip"], 1.0, 6.0, 3.0)
    params["clahe_grid"] = _clamp_int(params["clahe_grid"], 2, 16, 8)

    params["adapt_block"] = _clamp_int(params["adapt_block"], 3, 51, 21)
    if params["adapt_block"] % 2 == 0:
        params["adapt_block"] += 1
    params["adapt_c"] = _clamp_int(params["adapt_c"], 0, 30, 10)

    params["morph_ksize"] = _clamp_int(params["morph_ksize"], 3, 15, 5)
    if params["morph_ksize"] % 2 == 0:
        params["morph_ksize"] += 1
    params["close_iter"] = _clamp_int(params["close_iter"], 0, 5, 2)
    params["open_iter"] = _clamp_int(params["open_iter"], 0, 5, 1)
    params["min_area_ratio"] = _clamp_float(params["min_area_ratio"], 0.001, 0.2, 0.02)
    params["aspect_tolerance"] = _clamp_float(params["aspect_tolerance"], 0.05, 0.5, 0.2)

    return params


def _build_a4_debug_views(frame: np.ndarray, params, roi_bbox=None):
    roi = frame
    offset_x = 0
    offset_y = 0
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        x1 = max(0, min(int(x1), frame.shape[1] - 1))
        y1 = max(0, min(int(y1), frame.shape[0] - 1))
        x2 = max(0, min(int(x2), frame.shape[1]))
        y2 = max(0, min(int(y2), frame.shape[0]))
        if x2 - x1 > 1 and y2 - y1 > 1:
            roi = frame[y1:y2, x1:x2]
            offset_x = x1
            offset_y = y1

    blurred = cv2.GaussianBlur(
        roi,
        (params["blur_ksize"], params["blur_ksize"]),
        0,
    )
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(
        clipLimit=params["clahe_clip"],
        tileGridSize=(params["clahe_grid"], params["clahe_grid"]),
    )
    enhanced = clahe.apply(l_channel)

    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        params["adapt_block"],
        params["adapt_c"],
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (params["morph_ksize"], params["morph_ksize"]),
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=params["close_iter"])
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=params["open_iter"])

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = roi.shape[:2]
    min_area = params["min_area_ratio"] * (h * w)

    contours_view = roi.copy()
    cv2.drawContours(contours_view, contours, -1, (0, 255, 255), 1)

    overlay = roi.copy()
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
        if ratio_score < max(0.0, 1.0 - params["aspect_tolerance"]):
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
            "ratio": ratio,
            "area": area,
        }

    status = {
        "detected": False,
    }

    if best is not None:
        corners = best["rect"].astype(int)
        cv2.polylines(overlay, [corners], True, (0, 255, 0), 2)
        orientation = "portrait" if best["height_px"] >= best["width_px"] else "landscape"
        long_px = max(best["width_px"], best["height_px"])
        short_px = min(best["width_px"], best["height_px"])
        cm_per_px = ((A4_HEIGHT_CM / long_px) + (A4_WIDTH_CM / short_px)) * 0.5

        status = {
            "detected": True,
            "width_px": round(best["width_px"], 1),
            "height_px": round(best["height_px"], 1),
            "ratio": round(best["ratio"], 3),
            "area_ratio": round(best["area"] / float(h * w), 4),
            "orientation": orientation,
            "cm_per_px": round(cm_per_px, 5),
        }

    l_view = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2BGR)
    clahe_view = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    thresh_view = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    closed_view = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    opened_view = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)

    if offset_x or offset_y:
        canvas_shape = frame.shape
        def _paste(view):
            canvas = np.zeros(canvas_shape, dtype=view.dtype)
            canvas[offset_y:offset_y + view.shape[0], offset_x:offset_x + view.shape[1]] = view
            return canvas

        overlay = _paste(overlay)
        contours_view = _paste(contours_view)
        blurred = _paste(blurred)
        l_view = _paste(l_view)
        clahe_view = _paste(clahe_view)
        thresh_view = _paste(thresh_view)
        closed_view = _paste(closed_view)
        opened_view = _paste(opened_view)

        cv2.rectangle(
            overlay,
            (offset_x, offset_y),
            (offset_x + w, offset_y + h),
            (255, 255, 0),
            2,
        )

    return {
        "overlay": overlay,
        "blur": blurred,
        "l_channel": l_view,
        "clahe": clahe_view,
        "thresh": thresh_view,
        "closed": closed_view,
        "opened": opened_view,
        "contours": contours_view,
        "status": status,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/aruco")
def aruco_page():
    return render_template("aruco.html")


@app.route("/a4-debug")
def a4_debug_page():
    return render_template("a4_debug.html")


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

    if debug_streaming:
        emit("error", {"message": "Stop A4 debug before starting the main stream."})
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
    global cap, streaming, frame_count, person_bbox, is_cut_off, distance, focal_length, marker_corners, warning_message, all_persons, selected_person_idx, detection_mode, last_a4_detection

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
                if person_bbox is not None:
                    last_a4_detection = detect_a4_paper(
                        frame,
                        input_color="bgr",
                        roi_bbox=tuple(person_bbox),
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
            display_frame = draw_a4_overlay(display_frame, last_a4_detection)

        # Build status info
        status = {
            "detection_mode": detection_mode,
            "person_detected": person_bbox is not None,
            "is_cut_off": is_cut_off,
            "warning": warning_message,
            "aruco_detected": distance is not None if detection_mode == "aruco" else False,
            "distance": round(distance, 1) if distance and detection_mode == "aruco" else None,
            "a4_detected": last_a4_detection is not None if detection_mode == "a4" else False,
            "a4_cm_per_px": round(last_a4_detection.cm_per_px, 5) if last_a4_detection and detection_mode == "a4" else None,
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


@socketio.on("start_a4_debug")
def handle_start_a4_debug(data=None):
    global debug_cap, debug_streaming, a4_debug_params

    if debug_streaming:
        return
    if streaming:
        emit("a4_debug_error", {"message": "Stop the main stream before starting A4 debug."})
        return

    a4_debug_params = _sanitize_a4_debug_params(data or {})

    debug_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not debug_cap.isOpened():
        debug_cap = cv2.VideoCapture(0)
    if not debug_cap.isOpened():
        emit("a4_debug_error", {"message": "Could not open webcam."})
        return

    debug_streaming = True
    emit("a4_debug_started")
    socketio.start_background_task(a4_debug_loop)


@socketio.on("stop_a4_debug")
def handle_stop_a4_debug():
    global debug_streaming, debug_cap
    debug_streaming = False
    if debug_cap:
        debug_cap.release()
        debug_cap = None
    emit("a4_debug_stopped")


@socketio.on("update_a4_debug")
def handle_update_a4_debug(data):
    global a4_debug_params
    if not isinstance(data, dict):
        return
    merged = a4_debug_params.copy()
    merged.update(data)
    a4_debug_params = _sanitize_a4_debug_params(merged)


def a4_debug_loop():
    global debug_cap, debug_streaming, a4_debug_params, debug_last_person_bbox

    while debug_streaming and debug_cap is not None and debug_cap.isOpened():
        ret, frame = debug_cap.read()
        if not ret:
            break

        person_bbox = None
        if yolo_model is not None:
            results = yolo_model(frame, classes=[0], device=device, imgsz=320, conf=CONF_THRESHOLD, verbose=False)
            if len(results[0].boxes) > 0:
                h, w = frame.shape[:2]
                frame_center_x = w / 2
                frame_center_y = h / 2
                min_dist = float("inf")

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2
                    dist = math.hypot(box_center_x - frame_center_x, box_center_y - frame_center_y)
                    if dist < min_dist:
                        min_dist = dist
                        person_bbox = (x1, y1, x2, y2)

        if person_bbox is not None:
            debug_last_person_bbox = person_bbox
        else:
            person_bbox = debug_last_person_bbox

        debug_views = _build_a4_debug_views(frame, a4_debug_params, roi_bbox=person_bbox)

        socketio.emit(
            "a4_debug_frame",
            {
                "overlay": encode_frame(debug_views["overlay"]),
                "blur": encode_frame(debug_views["blur"]),
                "l_channel": encode_frame(debug_views["l_channel"]),
                "clahe": encode_frame(debug_views["clahe"]),
                "thresh": encode_frame(debug_views["thresh"]),
                "closed": encode_frame(debug_views["closed"]),
                "opened": encode_frame(debug_views["opened"]),
                "contours": encode_frame(debug_views["contours"]),
                "status": debug_views["status"],
            },
        )
        socketio.sleep(0.03)

    if debug_cap:
        debug_cap.release()
    debug_streaming = False


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

    a4_detection = detect_a4_paper(
        frame,
        input_color="bgr",
        roi_bbox=tuple(person_bbox),
    )
    if a4_detection is None:
        emit("capture_result", {"success": False, "error": "No A4 paper detected for scale calibration"})
        return

    height_est = estimate_human_height(
        frame,
        a4_detection,
        person_bbox=tuple(person_bbox),
        person_mask=mask,
    )

    if height_est is None:
        emit("capture_result", {"success": False, "error": "Failed to estimate height from A4 scale"})
        return

    result_frame = draw_a4_overlay(result_frame, a4_detection)

    if polygon is not None:
        highest_y = int(np.min(polygon[:, 1]))
        lowest_y = int(np.max(polygon[:, 1]))
        center_x = int(np.mean(polygon[:, 0]))
    else:
        x1, y1, x2, y2 = person_bbox
        highest_y, lowest_y = y1, y2
        center_x = int((x1 + x2) / 2)

    pixel_height = height_est.height_px

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
        "height_cm": round(height_est.height_cm, 1),
        "pixel_height": round(pixel_height, 1),
        "cm_per_px": round(a4_detection.cm_per_px, 5),
        "detection_mode": "a4",
    })


@socketio.on("disconnect")
def handle_disconnect():
    global streaming, cap, debug_streaming, debug_cap
    streaming = False
    if cap:
        cap.release()
        cap = None
    debug_streaming = False
    if debug_cap:
        debug_cap.release()
        debug_cap = None
    print("Client disconnected")


if __name__ == "__main__":
    load_models()
    print("\nStarting web app at http://localhost:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)
