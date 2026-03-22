import cv2
import cv2.aruco as aruco
import math
import numpy as np
import time
import base64
import torch
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO, SAM

app = Flask(__name__)
app.config["SECRET_KEY"] = "height-detection-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# --- Global State ---
yolo_model = None
sam_model = None
device = None
cap = None
streaming = False

# ArUco setup
MARKER_SIZE = 6  # cm
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
param_markers = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(marker_dict, param_markers)

# Detection state
EDGE_MARGIN = 15
frame_count = 0
skip_rate = 3
person_bbox = None
is_cut_off = False
warning_message = ""
distance = None
focal_length = None
marker_corners = None


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
def handle_start_stream():
    global cap, streaming, frame_count, person_bbox, is_cut_off, warning_message, distance, focal_length, marker_corners

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

    emit("stream_started")
    print("Stream started")

    # Run the streaming loop in a background thread so the handler returns
    socketio.start_background_task(stream_loop)


def stream_loop():
    global cap, streaming, frame_count, person_bbox, is_cut_off, distance, focal_length, marker_corners, warning_message

    while streaming and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        display_frame = frame.copy()
        frame_count += 1

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # YOLO + ArUco detection on every nth frame
        if frame_count % skip_rate == 0:
            results = yolo_model(frame, classes=[0], device=device, imgsz=320, verbose=False)

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

            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_bbox = [x1, y1, x2, y2]

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
                is_cut_off = False
                warning_message = ""

        # Draw bounding box
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            color = (0, 0, 255) if is_cut_off else (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

        # Draw ArUco markers
        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(display_frame, marker_corners)

        # Build status info
        status = {
            "person_detected": person_bbox is not None,
            "is_cut_off": is_cut_off,
            "warning": warning_message,
            "aruco_detected": distance is not None,
            "distance": round(distance, 1) if distance else None,
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
    global cap, streaming, person_bbox, is_cut_off, distance, focal_length

    if not streaming or cap is None or not cap.isOpened():
        emit("capture_result", {"success": False, "error": "Stream not active"})
        return

    if person_bbox is None:
        emit("capture_result", {"success": False, "error": "No person detected"})
        return

    if is_cut_off:
        emit("capture_result", {"success": False, "error": f"Cannot capture: {warning_message}"})
        return

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

    if sam_results[0].masks is not None and len(sam_results[0].masks.xy) > 0:
        polygon = sam_results[0].masks.xy[0]
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
        mask = sam_results[0].masks.data[0].cpu().numpy()
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
        })
    else:
        emit("capture_result", {"success": False, "error": "SAM failed to generate mask"})


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
    print("\nStarting web app at http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
