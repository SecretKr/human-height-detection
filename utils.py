import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import os

def find_corners(yolo_crop):
    """
    Takes a cropped image from YOLO and returns the 4 corners of the card.
    """
    print("Starting corner detection...")
    os.makedirs("results", exist_ok=True)

    # 1. Preprocessing: Grayscale and Blur to remove noise
    gray = cv2.cvtColor(yolo_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Edge Detection (Using Otsu's method to find optimal thresholds dynamically)
    high_thresh, thresh_im = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    canny_edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    # Optional: Dilate edges slightly to close any small gaps
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(canny_edges, kernel, iterations=1)
    
    # 3. Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (the card should be the largest object in the crop)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    print("Counturs found:", len(contours))
    if (len(contours) == 0):
        print("No contours found!")
        return None, None
    
    largest_contour = contours[0]
    
    # 4. Approximate the polygon
    perimeter = cv2.arcLength(largest_contour, True)
    # 2% of the perimeter is a good epsilon for polygon approximation
    approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
    print("Approximated points:", len(approx))  # , "|", approx)
    yolo_crop_copy = yolo_crop.copy()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 5, 1)
    plt.title("Original Crop")
    plt.imshow(cv2.cvtColor(yolo_crop, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 5, 2)
    plt.title("Blurred Grayscale")
    plt.imshow(gray, cmap='gray')
    plt.subplot(1, 5, 3)
    plt.title("Canny Edges")
    plt.imshow(canny_edges, cmap='gray')
    plt.subplot(1, 5, 4)
    plt.title("Dilated Edges")
    plt.imshow(edges, cmap='gray')
    plt.subplot(1, 5, 5)
    plt.title("Detected Card")
    cv2.drawContours(yolo_crop_copy, [approx], 0, (0, 255, 0), 1)
    plt.imshow(cv2.cvtColor(yolo_crop_copy, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.savefig("results/find_rect_result.png")
    plt.close()

    # Sanity Check: Ensure the contour isn't just tiny background noise
    crop_area = yolo_crop.shape[0] * yolo_crop.shape[1]
    if cv2.contourArea(largest_contour) < (crop_area * 0.15):
        return None, None # The largest contour is too small to be the card
        
    # 4. Get the tightly rotated bounding box
    # rect returns: (center(x, y), (width, height), angle)
    rect = cv2.minAreaRect(largest_contour)
    
    # Extract dimensions
    dim1, dim2 = rect[1]
    
    # The true pixel length is the longest side
    pixel_length = max(dim1, dim2)
    
    # We also return the rect so you can draw it for debugging
    return pixel_length, rect