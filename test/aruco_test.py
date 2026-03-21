import cv2
import cv2.aruco as aruco
import numpy as np
import math

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

# --- Configuration ---
MARKER_SIZE = 6  # centimeters
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
param_markers = aruco.DetectorParameters()
detector = aruco.ArucoDetector(marker_dict, param_markers)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = detector.detectMarkers(gray_frame)

    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv2.aruco.drawDetectedMarkers(frame, marker_corners)
            
            focal_length = 0.9 * frame.shape[1]
            center = (frame.shape[1] / 2, frame.shape[0] / 2)
            cam_mat = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coef = np.zeros((4, 1))
            
            # Estimate Pose
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, cam_mat, dist_coef
            )
            
            cv2.drawFrameAxes(frame, cam_mat, dist_coef, rvec, tvec, MARKER_SIZE / 2)

            # Rotation
            R, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = rotationMatrixToEulerAngles(R)
            r_x = math.degrees(pitch)
            r_y = math.degrees(yaw)
            r_z = math.degrees(roll)

            # --- FIXED SECTION ---
            # tvec shape is [[[x, y, z]]]
            t_x = tvec[0][0][0]
            t_y = tvec[0][0][1]
            t_z = tvec[0][0][2]
            
            distance = math.sqrt(t_x**2 + t_y**2 + t_z**2)

            text_pos = (int(corners[0][0][0]), int(corners[0][0][1]) - 20)
            status_text = f"x:{t_x:.1f} y:{t_y:.1f} z:{t_z:.1f} | Dist:{distance:.1f}cm"
            rot_text = f"Rx:{r_x:.0f} Ry:{r_y:.0f} Rz:{r_z:.0f}"

            cv2.putText(frame, status_text, text_pos, cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, rot_text, (text_pos[0], text_pos[1] + 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()