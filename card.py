import cv2
import math
import numpy as np
import time
from ultralytics import YOLO, SAM
import gdown
import os
from utils import find_corners

def main():
    print("Loading AI Models... (This may take a moment)")
    os.makedirs("results", exist_ok=True)
    
    if (not os.path.exists("yolo26n-card.pt")):
        gdown.download(id="1Hr5vKGWuItkoVsP--1ghqJ2sZiBsSNDL", output="yolo26n-card.pt")
    # 1. Base YOLO for person detection
    person_model = YOLO("yolo26n.pt")  
    # 2. Fine-tuned YOLO for card detection
    card_model = YOLO("yolo26n-card.pt")
    # 3. SAM for pixel-perfect edge segmentation
    sam_model = SAM("sam_b.pt")       

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera initialized at: {actual_w}x{actual_h}")

    # --- Application States ---
    STATE_LIVE = 0
    STATE_COUNTDOWN = 1
    STATE_RESULT = 2
    
    current_state = STATE_LIVE
    countdown_start_time = 0
    
    # Tracking variables
    person_bbox = None
    card_bbox = None
    capture_display = None
    
    # Standard ISO ID Card Dimensions (in cm)
    CARD_REAL_LENGTH = 8.56
    # CARD_REAL_WIDTH = 5.398

    # --- Optimization & Logic Variables ---
    EDGE_MARGIN = 15 
    frame_count = 0
    skip_rate = 3  # Run YOLO only every 3rd frame to save resources
    is_cut_off = False
    warning_message = ""

    print("\n--- Controls ---")
    print("Press 'c' to start 5-second capture countdown.")
    print("Press 'c' again to return to live view after capture.")
    print("Press 'q' to quit.")
    print("----------------\n")

    while cap.isOpened():
        if current_state in [STATE_LIVE, STATE_COUNTDOWN]:
            ret, frame = cap.read()
            if not ret:
                break
                
            h, w = frame.shape[:2]
            display_frame = frame.copy()
            frame_count += 1
            if (frame_count == 1): print("Frame size:", frame.shape)

            # --- 1. YOLO INFERENCE (FRAME SKIPPING) ---
            if frame_count % skip_rate == 0:
                # Detect Person
                person_results = person_model(frame, classes=[0], imgsz=320, verbose=False)
                # Detect Card (Assuming class 0 is 'card')
                card_results = card_model(frame, imgsz=1280, verbose=False)
                
                # Update Person BBox
                if len(person_results[0].boxes) > 0:
                    box = person_results[0].boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    person_bbox = [x1, y1, x2, y2]
                    
                    # Check if person is cut off by camera frame
                    if y1 <= EDGE_MARGIN:
                        is_cut_off = True
                        warning_message = "WARNING: Head is cut off!"
                    elif y2 >= h - EDGE_MARGIN:
                        is_cut_off = True
                        warning_message = "WARNING: Feet are cut off!"
                    else:
                        is_cut_off = False
                else:
                    person_bbox = None
                    is_cut_off = False

                # Update Card BBox
                if len(card_results[0].boxes) > 0:
                    c_box = card_results[0].boxes[0]
                    cx1, cy1, cx2, cy2 = map(int, c_box.xyxy[0].tolist())
                    card_bbox = [cx1, cy1, cx2, cy2]
                    confidence = float(c_box.conf[0])
                else:
                    card_bbox = None

            # --- 2. DRAWING LIVE BOUNDING BOXES ---
            if person_bbox is not None:
                x1, y1, x2, y2 = person_bbox
                color = (0, 0, 255) if is_cut_off else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                if is_cut_off:
                    cv2.putText(display_frame, warning_message, (w//2 - 150, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            if card_bbox is not None:
                cx1, cy1, cx2, cy2 = card_bbox
                cv2.rectangle(display_frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)
                cv2.putText(display_frame, "Card", (cx1, cy1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # --- LIVE STATE LOGIC ---
            if current_state == STATE_LIVE:
                if person_bbox is not None and card_bbox is not None and not is_cut_off:
                    cv2.putText(display_frame, "Target & Card Locked - Press 'c'", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Waiting for person AND card...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Height Detector", display_frame)

            # --- COUNTDOWN STATE LOGIC ---
            elif current_state == STATE_COUNTDOWN:
                elapsed_time = time.time() - countdown_start_time
                remaining_time = 5 - int(elapsed_time)

                if remaining_time > 0:
                    cv2.putText(display_frame, str(remaining_time), (w//2 - 40, h//2 + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
                    cv2.imshow("Height Detector", display_frame)
                
                else:
                    capture_display = frame.copy()
                    
                    if person_bbox is not None and card_bbox is not None and not is_cut_off:
                            
                        # Determine Orientation based on bounding box proportions
                        card_w = card_bbox[2] - card_bbox[0]
                        card_h = card_bbox[3] - card_bbox[1]
                        orientation = "Portrait" if card_h > card_w else "Landscape"
                        original_pixel_length = max(card_w, card_h)

                        cv2.rectangle(capture_display, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
                        cv2.putText(capture_display, f"Confidence: {confidence * 100:.1f}%, Format: {orientation}", (cx1, cy1 - 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        crop_pad = 0  
                        crop_x1 = max(0, card_bbox[0] - crop_pad)
                        crop_y1 = max(0, card_bbox[1] - crop_pad)
                        crop_x2 = min(w, card_bbox[2] + crop_pad)
                        crop_y2 = min(h, card_bbox[3] + crop_pad)
                        yolo_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                        cv2.imwrite("results/yolo_crop.jpg", yolo_crop)

                        pixel_length, rect = find_corners(yolo_crop)
                        print("Original pixel length from YOLO bbox:", original_pixel_length)
                        print("Pixel length of card from cv2.findContours:", round(pixel_length))

                        # --- SAM Processing for Person ---
                        px1, py1 = max(0, person_bbox[0] - 10), max(0, person_bbox[1] - 10)
                        px2, py2 = min(w, person_bbox[2] + 10), min(h, person_bbox[3] + 10)
                        
                        print("Capturing high-res frame and processing SAM masks...")
                        sam_person_results = sam_model(frame, bboxes=[[px1, py1, px2, py2]], verbose=False)
                            
                        if sam_person_results[0].masks is not None and len(sam_person_results[0].masks.xy) > 0:
                            person_polygon = sam_person_results[0].masks.xy[0]
                            highest_y = int(np.min(person_polygon[:, 1]))
                            lowest_y = int(np.max(person_polygon[:, 1]))
                            center_x = int(np.mean(person_polygon[:, 0]))
                            
                            pixel_height = lowest_y - highest_y

                            # Final Height Calculation
                            # estimated_height = (pixel_height * distance) / focal_length
                            estimated_height = (CARD_REAL_LENGTH * pixel_height) / pixel_length
                            print(f"Estimated Height: {estimated_height:.2f} cm")

                            # Draw person mask overlay
                            mask = sam_person_results[0].masks.data[0].cpu().numpy()
                            mask_resized = cv2.resize(mask.astype(np.uint8), (w, h))
                            overlay = capture_display.copy()
                            overlay[mask_resized > 0] = overlay[mask_resized > 0] * 0.6 + np.array([0, 120, 255]) * 0.4
                            capture_display = overlay   # .astype(np.uint8)
                                
                            # Draw Person Metrics
                            cv2.circle(capture_display, (center_x, highest_y), 5, (0, 0, 255), -1)
                            cv2.circle(capture_display, (center_x, lowest_y), 5, (0, 0, 255), -1)
                            cv2.line(capture_display, (center_x, highest_y), (center_x, lowest_y), (255, 0, 0), 2)
                                
                            cv2.putText(capture_display, f"Height: {estimated_height:.2f} cm", (10, 40), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                        else:
                            cv2.putText(capture_display, "SAM Error: Person mask failed", (10, 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # else:
                    #     cv2.putText(capture_display, "SAM Error: Card mask failed", (10, 40), 
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(capture_display, "Capture failed: Missing Target or Card", (10, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.putText(capture_display, "Press 'c' to return to live feed", (10, h - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    current_state = STATE_RESULT

        # --- RESULT STATE LOGIC ---
        elif current_state == STATE_RESULT:
            cv2.imshow("Height Detector", capture_display)

        # --- KEYBOARD CONTROLS ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if current_state == STATE_LIVE:
                if person_bbox is not None and card_bbox is not None:
                    current_state = STATE_COUNTDOWN
                    countdown_start_time = time.time()
                else:
                    print("Cannot start countdown: Ensure both person and card are visible!")
            
            elif current_state == STATE_RESULT:
                print("Returning to live view...")
                current_state = STATE_LIVE
                
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()