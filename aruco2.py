import cv2
import cv2.aruco as aruco
import numpy as np
import math

# ArUco setup (from your code)
MARKER_SIZE = 6  # cm
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
param_markers = aruco.DetectorParameters()
detector = aruco.ArucoDetector(marker_dict, param_markers)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    marker_corners, marker_IDs, reject = detector.detectMarkers(gray_frame)

    top_id = 1
    bottom_id = 2

    top_tvec = None
    bottom_tvec = None

    if marker_IDs is not None:
        pixel_size = np.linalg.norm(marker_corners[0][0][0] - marker_corners[0][0][1])
        for ids, corners in zip(marker_IDs, marker_corners):
            focal_length = 0.9 * frame.shape[1]
            center = (frame.shape[1] / 2, frame.shape[0] / 2)
            cam_mat = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coef = np.zeros((4, 1))
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cam_mat, dist_coef)
            cv2.aruco.drawDetectedMarkers(frame, marker_corners)

        for i, marker_id in enumerate(marker_IDs.flatten()):
            if marker_id == top_id:
                top_tvec = tvec[i][0]
            elif marker_id == bottom_id:
                bottom_tvec = tvec[i][0]

    if top_tvec is not None and bottom_tvec is not None:  
        height = np.linalg.norm(top_tvec - bottom_tvec)
        cv2.putText(frame, f"Height: {height:.1f} cm", (10, 30), 
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Height Estimation", frame)
    if cv2.waitKey(1) == ord("q"):
        break
            

cap.release()
cv2.destroyAllWindows()