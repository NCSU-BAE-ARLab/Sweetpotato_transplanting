import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
SEGMENT_COLOR = (0, 255, 0)       # Green for masks
KEYPOINT_COLOR_TAIL = (0, 0, 255)      # Red for visible keypoints
KEYPOINT_COLOR_HEAD = (255, 0, 0)      # Red for visible keypoints
KEYPOINT_INVIS_COLOR = (255, 255, 0)  # Yellow for occluded keypoints
KEYPOINT_RADIUS = 5


data_root = "validation_data_annotated/leaf/overlap/"

# --- Load COCO JSON annotations ---
with open(data_root+"coco_leaf_overlap.json", "r") as f:
    coco_data = json.load(f)

# --- Load image ---
image_info = coco_data["images"][3]
image_path = data_root+image_info["file_name"]
image = cv2.imread(image_path)
cv2.imwrite('show_image.png',image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Get annotations for the image ---
annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_info["id"]]

for ann in annotations:
    

    # --- Draw segmentation mask ---
    for seg in ann["segmentation"]:
        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=SEGMENT_COLOR, thickness=2)


    if ann["category_id"] != 1:
        continue
    # --- Draw keypoints ---
    keypoints = ann.get("keypoints", [])
    for i in range(0, len(keypoints), 3):
        if i:
            x, y, v = keypoints[i:i+3]
            if v == 2:
                cv2.circle(image, (int(x), int(y)), KEYPOINT_RADIUS, KEYPOINT_COLOR_TAIL, -1)
            elif v == 1:
                cv2.circle(image, (int(x), int(y)), KEYPOINT_RADIUS, KEYPOINT_COLOR_TAIL, 2)
        else:
            x, y, v = keypoints[i:i+3]
            if v == 2:
                cv2.circle(image, (int(x), int(y)), KEYPOINT_RADIUS, KEYPOINT_COLOR_HEAD, -1)
            elif v == 1:
                cv2.circle(image, (int(x), int(y)), KEYPOINT_RADIUS, KEYPOINT_COLOR_HEAD, 2)

# --- Save output image ---
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.savefig("show.png", bbox_inches="tight", pad_inches=0)
plt.close()
