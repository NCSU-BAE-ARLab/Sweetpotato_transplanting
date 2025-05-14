import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2
import json
import os
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.ops import box_iou

import matplotlib.pyplot as plt

# ====================
# 1. Model loading
# ====================
# Load model
def get_model(num_keypoints=2,num_classes=2,weights_path=None, device="cuda"):
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
    # MaskRCNNPredictor()
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



# ====================
# 2. COCO + Transforms
# ====================
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image), np.array(image)

# ====================
# 3. IoU calculator
# ====================
def compute_mask_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

# ====================
# 4. Validation routine
# ====================
def validate(model, coco_json_path, image_folder, device="cuda", score_thresh=0.5, visualize=False):
    coco = COCO(coco_json_path)
    image_ids = coco.getImgIds()

    all_ious = []
    all_precisions = []

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(image_folder, img_info["file_name"])
        image_tensor, image_np = load_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)[0]

        pred_masks = output["masks"] > 0.5
        pred_scores = output["scores"]
        pred_masks = pred_masks[pred_scores > score_thresh]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        gt_masks = []
        for ann in anns:
            rle = coco.annToRLE(ann)
            m = coco_mask.decode(rle)
            gt_masks.append(m)

        matched_gt = set()
        ious = []
        for pm in pred_masks:
            pm_np = pm.squeeze().cpu().numpy()
            best_iou = 0
            for idx, gm in enumerate(gt_masks):
                if idx in matched_gt:
                    continue
                gm_np = gm.squeeze()
                iou = compute_mask_iou(pm_np, gm_np)
                if iou > best_iou:
                    best_iou = iou
                    matched_gt.add(idx)
            ious.append(best_iou)

        if ious:
            all_ious.extend(ious)
            all_precisions.append(np.mean([iou >= 0.5 for iou in ious]))

        if visualize:
            overlay = image_np.copy()
            for pm in pred_masks:
                mask = pm.squeeze().cpu().numpy().astype(np.uint8)
                color = np.random.randint(0, 255, 3)
                overlay[mask > 0] = 0.5 * overlay[mask > 0] + 0.5 * color
            plt.imshow(overlay)
            plt.title(f"Image {img_id}")
            plt.axis("off")
            plt.show()

    mean_iou = np.mean(all_ious) if all_ious else 0
    mean_precision = np.mean(all_precisions) if all_precisions else 0

    print(f"\n Validation Results:\n---------------------")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"mAP@0.5 (approx): {mean_precision:.4f}")

# ====================
# Run
# ====================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 2  # 1 class + background
    weights_path = "../results_blackB_old_slips/maskrcnn_slip_10.pth"

    data_root = '../dataset_generation_sps/validation_data_annotated/non-leaf/non-overlap/'

    coco_json_path = data_root + "coco.json"  # or full path
    image_folder = data_root + "./"

    model = get_model(num_classes=num_classes, weights_path=weights_path, device=device)
    validate(model, coco_json_path, image_folder, device=device, visualize=False)
