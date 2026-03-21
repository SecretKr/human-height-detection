import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Setup Model
checkpoint = "sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def get_height_from_mask(mask):
    # Find the topmost and bottommost pixel of the mask
    coords = np.argwhere(mask)
    y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
    return y_max - y_min

# 2. Load Image
image_path = "person_with_reference.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# 3. Define Points (X, Y) 
# Point 1: On the person | Point 2: On the reference object
input_points = np.array([[500, 1000], [200, 1000]]) 
input_labels = np.array([1, 1]) 

# 4. Predict Masks
masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True # Helps separate distinct objects
)

# Assume mask[0] is person, mask[1] is reference (or process individually)
person_pixel_height = get_height_from_mask(masks[0])
ref_pixel_height = get_height_from_mask(masks[1])

# 5. Conversion (Example: Reference object is 180cm tall)
KNOWN_REF_HEIGHT_CM = 180 
pixel_to_cm_ratio = KNOWN_REF_HEIGHT_CM / ref_pixel_height
estimated_height = person_pixel_height * pixel_to_cm_ratio

print(f"Estimated Height: {estimated_height:.2f} cm")