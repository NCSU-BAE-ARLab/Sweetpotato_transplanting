import torch
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
import torchvision
from PIL import Image
import torchvision.ops as ops
import copy


def get_model(num_keypoints=2,weights_path=None, device="cuda",num_classes=2):
    model  = maskrcnn_resnet50_fpn(pretrained=True)

    # Bbox and cls head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # 1 class + background
    
    #Keypoint head
    model.roi_heads.keypoint_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
    keypoint_layers = tuple(512 for _ in range(8))

    out_channels = model.backbone.out_channels
    model.roi_heads.keypoint_head = KeypointRCNNHeads(out_channels, keypoint_layers)

    keypoint_dim_reduced = 512  # == keypoint_layers[-1]
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced, num_keypoints)

    # Mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded model weights from {weights_path}")

    model.to(device).eval()

    return model

# Preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image), np.array(image)  # Tensor and numpy format

# Draw predictions on image
def draw_predictions(image_np, outputs, score_thresh=0.85):
    KEYPOINT_COLOR_HEAD = (255, 0, 0)   # Blue
    KEYPOINT_COLOR_TAIL = (0, 0, 255)   # Red
    KEYPOINT_RADIUS = 5
    SEGMENT_COLOR = (0, 255, 0)         # Green for mask boundary

    image = image_np.copy()

    for i in range(len(outputs["scores"])):
        score = outputs["scores"][i].item()
        if score < score_thresh:
            continue

        # --- Get mask ---
        if "masks" not in outputs or outputs["masks"] is None:
            continue
        full_mask = outputs["masks"][i, 0].cpu().numpy()
        binary_mask = (full_mask > 0.5).astype(np.uint8)

        # --- Keep only the largest connected component ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels <= 1:
            continue  # Skip if no connected component found
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned_mask = (labels == largest_component).astype(np.uint8)

        # --- Draw mask boundary (contour) ---
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, SEGMENT_COLOR, thickness=2)

        # --- Draw keypoints if inside the cleaned mask ---
        if "keypoints" in outputs and outputs["keypoints"] is not None:
            keypoints = outputs["keypoints"][i].cpu().numpy()
            for j, (x, y, v) in enumerate(keypoints):
                if v > 0 and 0 <= int(x) < cleaned_mask.shape[1] and 0 <= int(y) < cleaned_mask.shape[0]:
                    if cleaned_mask[int(y), int(x)] == 1:
                        color = KEYPOINT_COLOR_HEAD if j == 0 else KEYPOINT_COLOR_TAIL
                        thickness = -1 #if v == 2 else 2
                        cv2.circle(image, (int(x), int(y)), KEYPOINT_RADIUS, color, thickness)

        # Optional: draw score
        box = outputs["boxes"][i].cpu().numpy().astype(int)
        center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        cv2.putText(image, f"{score:.4f}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image

# Run inference
def main(image_path, weights_path="model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(weights_path=weights_path).to(device)

    image_tensor, image_np = load_image(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    iou_threshold = 0.5
    # Apply NMS
    keep_indices = ops.nms(outputs["boxes"], outputs["scores"], iou_threshold)
    print(outputs.keys())
    # Filter outputs based on NMS
    outputs = {
        "boxes": outputs["boxes"][keep_indices],
        "scores": outputs["scores"][keep_indices],
        "masks": outputs["masks"][keep_indices] if "masks" in outputs else None,
        "keypoints": outputs["keypoints"][keep_indices] if "keypoints" in outputs else None,
        "labels": outputs["labels"][keep_indices],
    }


    # Draw results
    result_image = draw_predictions(image_np.copy(), outputs)

    # Show image
    plt.figure(figsize=(8, 8))
    plt.imshow(result_image)
    plt.axis("off")
    #plt.show()
    plt.savefig('../results'+f"/predictions_single_whole.png")

# Example Usage
if __name__ == "__main__":
    main("../dataset_generation_sps/0_ref_image.png", "../results_blackB_old_slips/maskrcnn_slip_1.pth")  # Replace with your image and model file
