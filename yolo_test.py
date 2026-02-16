import cv2
import math
from ultralytics import YOLO

def detect_and_measure():
    # Load the YOLOv8 model
    # 'yolov8n.pt' is the fastest (nano). Use 'yolov8m.pt' for more precision.
    model = YOLO('yolov8n.pt') 

    # Open webcam (change to 'video.mp4' to use a file)
    cap = cv2.VideoCapture(0)

    # Set camera resolution (optional, higher = more precision but slower)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference on the frame
        # classes=0 tells YOLO to only look for 'person' (class 0)
        results = model(frame, classes=0, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box Coordinates (x1, y1, x2, y2)
                # y1 is the top (head), y2 is the bottom (toe)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Calculate Height in Pixels
                height_px = y2 - y1

                # --- DRAWING ---
                # Draw the square (bounding box)
                # Color: (0, 255, 0) = Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Draw a line specifically from Head to Toe for visualization
                cv2.line(frame, (int((x1+x2)/2), y1), (int((x1+x2)/2), y2), (0, 0, 255), 2)

                # Display the pixel height
                cv2.putText(frame, f'Height: {height_px} px', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Human Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_measure()