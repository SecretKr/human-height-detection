import cv2
import cv2.aruco as aruco
import math
import numpy as np
import time
import base64
import os
import torch
import gdown
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO, SAM

app = Flask(__name__)
app.config["SECRET_KEY"] = "height-detection-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# --- Global State ---
yolo_model = None
card_model = None
sam_model = None
device = None
cap = None
streaming = False
detection_mode = "aruco"  # "aruco" or "card"

# ArUco setup
MARKER_SIZE = 6  # cm
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
param_markers = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(marker_dict, param_markers)

# Detection state
CONF_THRESHOLD = 0.6
EDGE_MARGIN = 15
CARD_REAL_LENGTH = 8.56  # cm — standard ISO ID card
frame_count = 0
skip_rate = 3
person_bbox = None
is_cut_off = False
warning_message = ""
distance = None
focal_length = None
marker_corners = None
all_persons = []
selected_person_idx = -1
card_bbox = None


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
    global yolo_model, card_model, sam_model, device
    device = get_device()
    print(f"Using device: {device}")
    print("Loading YOLO and SAM models...")
    yolo_model = YOLO("yolo26n.pt")
    sam_model = SAM("sam_b.pt")
    # Load card detection model (download if not present)
    card_model_path = "yolo26n-card.pt"
    if not os.path.exists(card_model_path):
        print("Downloading card detection model...")
        gdown.download(id="1Hr5vKGWuItkoVsP--1ghqJ2sZiBsSNDL", output=card_model_path)
    card_model = YOLO(card_model_path)
    print("Models loaded successfully.")


def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode("utf-8")


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


@app.route("/api/set-mode", methods=["POST"])
def set_mode():
    global detection_mode, card_bbox, distance, focal_length, marker_corners
    data = request.json
    mode = data.get("mode") if data else None
    if mode not in ("aruco", "card"):
        return jsonify({"error": "Invalid mode. Use 'aruco' or 'card'."}), 400
    detection_mode = mode
    # Reset reference detection state
    card_bbox = None
    distance = None
    focal_length = None
    marker_corners = None
    return jsonify({"mode": detection_mode})


@socketio.on("start_stream")
def handle_start_stream():
    global cap, streaming, frame_count, person_bbox, is_cut_off, warning_message, distance, focal_length, marker_corners, all_persons, selected_person_idx, card_bbox

    if streaming:
        return

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
    all_persons = []
    selected_person_idx = -1
    card_bbox = None

    emit("stream_started")
    print("Stream started")

    # Run the streaming loop in a background thread so the handler returns
    socketio.start_background_task(stream_loop)


def stream_loop():
    global cap, streaming, frame_count, person_bbox, is_cut_off, distance, focal_length, marker_corners, warning_message, all_persons, selected_person_idx, card_bbox

    while streaming and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        display_frame = frame.copy()
        frame_count += 1

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # YOLO + reference marker detection on every nth frame
        if frame_count % skip_rate == 0:
            # Person detection (always at imgsz=320)
            results = yolo_model(frame, classes=[0], device=device, imgsz=320, conf=CONF_THRESHOLD, verbose=False)

            if detection_mode == "aruco":
                # Detect ArUco markers
                detected_corners, marker_IDs, _ = aruco_detector.detectMarkers(gray_frame)
                marker_corners = detected_corners

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

            elif detection_mode == "card":
                # Detect ID card at high resolution
                card_results = card_model(frame, imgsz=1280, verbose=False)
                if len(card_results[0].boxes) > 0:
                    c_box = card_results[0].boxes[0]
                    cx1, cy1, cx2, cy2 = map(int, c_box.xyxy[0].tolist())
                    card_bbox = [cx1, cy1, cx2, cy2]
                else:
                    card_bbox = None

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

        # Draw all bounding boxes
        for i, bbox in enumerate(all_persons):
            x1, y1, x2, y2 = bbox
            if i == selected_person_idx:
                color = (0, 0, 255) if is_cut_off else (0, 255, 0)
            else:
                color = (0, 255, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            # Label each person with index in card mode (multiple persons)
            if detection_mode == "card" and len(all_persons) > 1:
                cv2.putText(display_frame, f"P{i+1}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Draw ArUco markers (aruco mode only)
        if detection_mode == "aruco" and marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(display_frame, marker_corners)

        # Draw card bbox (card mode only)
        if detection_mode == "card" and card_bbox is not None:
            cx1, cy1, cx2, cy2 = card_bbox
            cv2.rectangle(display_frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
            cv2.putText(display_frame, "Card", (cx1, cy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Build status info
        status = {
            "mode": detection_mode,
            "person_detected": person_bbox is not None,
            "person_count": len(all_persons),
            "is_cut_off": is_cut_off,
            "warning": warning_message,
            # ArUco
            "aruco_detected": distance is not None,
            "distance": round(distance, 1) if distance else None,
            # Card
            "card_detected": card_bbox is not None,
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
    global cap, streaming, person_bbox, is_cut_off, distance, focal_length, all_persons, card_bbox

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
        _capture_aruco()
    elif detection_mode == "card":
        _capture_card()


def _capture_aruco():
    """Capture and measure height using ArUco marker as reference (single person)."""
    if distance is None or focal_length is None:
        emit("capture_result", {"success": False, "error": "No ArUco marker detected for distance calibration"})
        return

    ret, frame = cap.read()
    if not ret:
        emit("capture_result", {"success": False, "error": "Failed to capture frame"})
        return

    h, w = frame.shape[:2]
    pad = 10
    px1, py1 = max(0, person_bbox[0] - pad), max(0, person_bbox[1] - pad)
    px2, py2 = min(w, person_bbox[2] + pad), min(h, person_bbox[3] + pad)

    try:
        sam_results = sam_model(frame, bboxes=[[px1, py1, px2, py2]], device=device, verbose=False)
    except Exception as e:
        emit("capture_result", {"success": False, "error": f"SAM error: {str(e)}"})
        return

    result_frame = frame.copy()

    if sam_results[0].masks is not None and len(sam_results[0].masks.xy) > 0:
        polygon = sam_results[0].masks.xy[0]
        highest_y = int(np.min(polygon[:, 1]))
        lowest_y = int(np.max(polygon[:, 1]))
        center_x = int(np.mean(polygon[:, 0]))

        pixel_height = lowest_y - highest_y
        estimated_height = (pixel_height * distance) / focal_length

        # Draw mask overlay
        mask = sam_results[0].masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))
        overlay = result_frame.copy()
        overlay[mask_resized > 0] = overlay[mask_resized > 0] * 0.6 + np.array([0, 120, 255]) * 0.4
        result_frame = overlay.astype(np.uint8)

        # Draw measurement lines on top of overlay
        cv2.circle(result_frame, (center_x, highest_y), 7, (0, 0, 255), -1)
        cv2.circle(result_frame, (center_x, lowest_y), 7, (0, 0, 255), -1)
        cv2.line(result_frame, (center_x, highest_y), (center_x, lowest_y), (255, 255, 0), 2)

        frame_data = encode_frame(result_frame)
        emit("capture_result", {
            "success": True,
            "mode": "aruco",
            "image": frame_data,
            "height_cm": round(estimated_height, 1),
            "pixel_height": pixel_height,
            "distance_cm": round(distance, 1),
        })
    else:
        emit("capture_result", {"success": False, "error": "SAM failed to generate mask"})


def _capture_card():
    """Capture and measure height for ALL detected persons using ID card as reference."""
    if card_bbox is None:
        emit("capture_result", {"success": False, "error": "No ID card detected. Hold a standard ID card in the scene."})
        return

    ret, frame = cap.read()
    if not ret:
        emit("capture_result", {"success": False, "error": "Failed to capture frame"})
        return

    h, w = frame.shape[:2]

    # Compute card pixel length from tracked bbox
    card_w = card_bbox[2] - card_bbox[0]
    card_h = card_bbox[3] - card_bbox[1]
    card_pixel_length = max(card_w, card_h)

    if card_pixel_length < 10:
        emit("capture_result", {"success": False, "error": "Card bounding box too small, reposition card"})
        return

    persons_to_process = all_persons if all_persons else ([person_bbox] if person_bbox else [])

    result_frame = frame.copy()

    # Draw card bbox on result
    cx1, cy1, cx2, cy2 = card_bbox
    cv2.rectangle(result_frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
    cv2.putText(result_frame, "Card", (cx1, cy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    person_colors = [(255, 120, 0), (0, 120, 255), (120, 255, 0), (255, 0, 180)]
    results_list = []

    for i, bbox in enumerate(persons_to_process):
        pad = 10
        px1, py1 = max(0, bbox[0] - pad), max(0, bbox[1] - pad)
        px2, py2 = min(w, bbox[2] + pad), min(h, bbox[3] + pad)

        try:
            sam_results = sam_model(frame, bboxes=[[px1, py1, px2, py2]], device=device, verbose=False)
        except Exception:
            continue

        if sam_results[0].masks is None or len(sam_results[0].masks.xy) == 0:
            continue

        polygon = sam_results[0].masks.xy[0]
        highest_y = int(np.min(polygon[:, 1]))
        lowest_y = int(np.max(polygon[:, 1]))
        center_x = int(np.mean(polygon[:, 0]))

        pixel_height = lowest_y - highest_y
        estimated_height = (CARD_REAL_LENGTH * pixel_height) / card_pixel_length

        results_list.append({
            "person_idx": i + 1,
            "height_cm": round(estimated_height, 1),
            "pixel_height": pixel_height,
        })

        # Draw mask overlay with per-person colour
        color = person_colors[i % len(person_colors)]
        mask = sam_results[0].masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))
        overlay = result_frame.copy()
        overlay[mask_resized > 0] = overlay[mask_resized > 0] * 0.6 + np.array(color) * 0.4
        result_frame = overlay.astype(np.uint8)

        # Draw measurement lines
        cv2.circle(result_frame, (center_x, highest_y), 7, (0, 0, 255), -1)
        cv2.circle(result_frame, (center_x, lowest_y), 7, (0, 0, 255), -1)
        cv2.line(result_frame, (center_x, highest_y), (center_x, lowest_y), (255, 255, 0), 2)
        cv2.putText(result_frame, f"P{i+1}: {estimated_height:.1f} cm",
                    (center_x + 10, (highest_y + lowest_y) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if results_list:
        frame_data = encode_frame(result_frame)
        emit("capture_result", {
            "success": True,
            "mode": "card",
            "image": frame_data,
            "persons": results_list,
            "card_pixel_length": card_pixel_length,
        })
    else:
        emit("capture_result", {"success": False, "error": "SAM failed to generate masks for any person"})


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
    socketio.run(app, host="127.0.0.1", port=5001, debug=False)
