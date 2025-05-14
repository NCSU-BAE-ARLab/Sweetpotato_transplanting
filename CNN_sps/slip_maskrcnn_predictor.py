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


class Predictor:
    def __init__(self,weights_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model(weights_path=weights_path).to(self.device)
        self.transform =  T.Compose([T.ToTensor()]) 
        
    def predict(self,image):
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            outputs = self.model(image_tensor)[0]

        # Draw results
        result_image, combine_masks, combine_keypoints = draw_predictions(np.array(image).copy(), outputs)

        return result_image, combine_masks, combine_keypoints
                     


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

# def get_model(num_keypoints=2,weights_path=None, device="cuda"):
#     model  = maskrcnn_resnet50_fpn(pretrained=True)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)  # 1 class + background
    
#     #Keypoint head
#     model.roi_heads.keypoint_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
#     keypoint_layers = tuple(512 for _ in range(8))

#     out_channels = model.backbone.out_channels
#     model.roi_heads.keypoint_head = KeypointRCNNHeads(out_channels, keypoint_layers)

#     keypoint_dim_reduced = 512  # == keypoint_layers[-1]
#     model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced, num_keypoints)

#     if weights_path:
#         model.load_state_dict(torch.load(weights_path, map_location=device))
#         print(f"Loaded model weights from {weights_path}")

#     model.to(device).eval()

#     return model

# Load the trained model
# def get_model(num_keypoints=2, weights_path="model.pth"):
#     model = maskrcnn_resnet50_fpn(pretrained=False)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)  # 1 class + background
#     model.roi_heads.keypoint_predictor = torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(in_features, num_keypoints)

#     model.load_state_dict(torch.load(weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
#     model.eval()
#     return model

# Preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image), np.array(image)  # Tensor and numpy format

# Draw predictions on image
def draw_predictions(image_np, outputs, score_thresh=0.5):

    overlay_color = np.array([255, 0, 255], dtype=np.uint8)  # Green color
    alpha = 0.25  # Transparency factor (0 = fully transparent, 1 = solid color)
    color_mask = np.zeros_like(image_np, dtype=np.uint8)
    combine_masks = []
    combine_keypoints = []
    print('total_predictions',len(outputs["scores"]))
    for i in range(len(outputs["scores"])):
        score = outputs["scores"][i].item()
        if score < score_thresh:
            print('Skipped. Score:',score)
            continue
        else:
            print('In. Score:',score)


        # Get bounding box
        # box = outputs["boxes"][i].cpu().numpy().astype(int)
        # cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Get mask
        mask = outputs["masks"][i].cpu().numpy().squeeze() > score_thresh
        # image_np[mask] = [0, 255, 0]  # Green overlay
        color_mask[mask] = overlay_color
        combine_masks.append(mask)

        # Get keypoints
        if "keypoints" in outputs:
            keypoints = outputs["keypoints"][i].cpu().numpy()
            combine_keypoints.append(keypoints)
            print('keypoints',keypoints)
            for j, (x, y, v) in enumerate(keypoints):
                if v > 0:  # Visibility check
                    if j:
                        cv2.circle(image_np, (int(x), int(y)), 4, (0, 255, 0), -1)
                    else:
                        cv2.circle(image_np, (int(x), int(y)), 4, (0, 0, 255), -1)

    image_np = cv2.addWeighted(image_np, 1 - alpha, color_mask, alpha, 0)
    combine_masks = np.stack(combine_masks, axis=0) # n x w x h
    combine_keypoints = np.stack(combine_keypoints, axis=0) # n x num_keypoints x 3

    return image_np, combine_masks, combine_keypoints

# Run inference
def main(image_path, weights_path="model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(weights_path=weights_path).to(device)

    image_tensor, image_np = load_image(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    # Draw results
    result_image = draw_predictions(image_np.copy(), outputs)

    # Show image
    plt.figure(figsize=(8, 8))
    plt.imshow(result_image)
    plt.axis("off")
    plt.show()
    plt.savefig('../results'+f"/predictions_single.png")

# Example Usage
if __name__ == "__main__":
    main("../dataset_generation_sps/sample_short_Color_640x480_Color.png", "../results/maskrcnn_slip_9.pth")  # Replace with your image and model file
