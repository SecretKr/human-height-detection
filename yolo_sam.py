import cv2
import numpy as np
import time
from ultralytics import YOLO, SAM

def main():
    print("Loading YOLOv26 and SAM models...")
    yolo_model = YOLO("yolo26n.pt")  
    sam_model = SAM("sam_b.pt")       

    cap = cv2.VideoCapture(0)

    # --- Application States ---
    STATE_LIVE = 0
    STATE_COUNTDOWN = 1
    STATE_RESULT = 2
    
    current_state = STATE_LIVE
    countdown_start_time = 0
    person_bbox = None
    capture_display = None
    
    # --- Optimization & Logic Variables ---
    EDGE_MARGIN = 15 
    frame_count = 0
    skip_rate = 3  # Run YOLO only every 3rd frame
    is_cut_off = False
    warning_message = ""

    print("\n--- Controls ---")
    print("Press 'c' to start 3-second capture countdown.")
    print("Press 'c' again to return to live view after capture.")
    print("Press 'q' to quit.")
    print("----------------\n")

    while cap.isOpened():
        if current_state in [STATE_LIVE, STATE_COUNTDOWN]:
            ret, frame = cap.read()
            if not ret:
                break
                
            h, w = frame.shape[:2] # Original High-Res dimensions
            display_frame = frame.copy()
            frame_count += 1
            
            # --- 1. OPTIMIZED YOLO INFERENCE (FRAME SKIPPING & LOWER RES) ---
            if frame_count % skip_rate == 0:
                # device='mps' uses Mac GPU, imgsz=320 processes a smaller image
                results = yolo_model(frame, classes=[0], device='mps', imgsz=320, verbose=False)
                
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    # Coordinates are automatically scaled back to high-res by Ultralytics
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    person_bbox = [x1, y1, x2, y2]
                    
                    # Check boundaries
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

            # --- 2. DRAWING THE BOUNDING BOX ---
            # We draw this every frame using the last known coordinates, keeping it smooth
            if person_bbox is not None:
                x1, y1, x2, y2 = person_bbox
                color = (0, 0, 255) if is_cut_off else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                if is_cut_off:
                    cv2.putText(display_frame, warning_message, (w//2 - 150, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            # --- LIVE STATE LOGIC ---
            if current_state == STATE_LIVE:
                if person_bbox is not None and not is_cut_off:
                    cv2.putText(display_frame, "Target Locked - Press 'c'", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif person_bbox is None:
                    cv2.putText(display_frame, "Waiting for person...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Height Detector", display_frame)

            # --- COUNTDOWN STATE LOGIC ---
            elif current_state == STATE_COUNTDOWN:
                elapsed_time = time.time() - countdown_start_time
                remaining_time = 3 - int(elapsed_time)

                if remaining_time > 0:
                    cv2.putText(display_frame, str(remaining_time), (w//2 - 40, h//2 + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
                    cv2.imshow("Height Detector", display_frame)
                
                else:
                    if person_bbox is not None:
                        if is_cut_off:
                            capture_display = frame.copy()
                            cv2.putText(capture_display, "ERROR: Capture aborted.", (10, 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            cv2.putText(capture_display, warning_message, (10, 80), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        else:
                            print("\n📸 Capturing high-res frame and processing SAM mask...")
                            pad = 10
                            px1, py1 = max(0, person_bbox[0] - pad), max(0, person_bbox[1] - pad)
                            px2, py2 = min(w, person_bbox[2] + pad), min(h, person_bbox[3] + pad)
                            padded_bbox = [[px1, py1, px2, py2]] 

                            # SAM processes the original HIGH-RES 'frame', not a downscaled one
                            sam_results = sam_model(frame, bboxes=padded_bbox, device='mps', verbose=False)
                            capture_display = frame.copy()
                            
                            if sam_results[0].masks is not None and len(sam_results[0].masks.xy) > 0:
                                polygon = sam_results[0].masks.xy[0]
                                highest_y = int(np.min(polygon[:, 1]))
                                lowest_y = int(np.max(polygon[:, 1]))
                                center_x = int(np.mean(polygon[:, 0]))
                                
                                pixel_height = lowest_y - highest_y
                                
                                cv2.circle(capture_display, (center_x, highest_y), 5, (0, 0, 255), -1)
                                cv2.circle(capture_display, (center_x, lowest_y), 5, (0, 0, 255), -1)
                                cv2.line(capture_display, (center_x, highest_y), (center_x, lowest_y), (255, 0, 0), 2)
                                
                                cv2.putText(capture_display, f"Height: {pixel_height} px", (10, 40), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                            else:
                                cv2.putText(capture_display, "SAM Error: No mask generated", (10, 40), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        capture_display = frame.copy()
                        cv2.putText(capture_display, "Capture failed: Target lost", (10, 40), 
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
                if person_bbox is not None:
                    current_state = STATE_COUNTDOWN
                    countdown_start_time = time.time()
                else:
                    print("⚠️ Cannot start countdown: No person detected!")
            
            elif current_state == STATE_RESULT:
                print("Returning to live view...")
                current_state = STATE_LIVE
                
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()