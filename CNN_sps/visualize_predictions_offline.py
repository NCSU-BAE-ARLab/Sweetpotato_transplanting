import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from dataset import CocoDataset  # Ensure you have your dataset class
import torchvision


from torchvision.transforms import functional as F, Normalize

# Define the same transform used in training
transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def plot_mask_histogram(mask, title="Mask Value Distribution"):
    """
    Plots the histogram of a given mask's pixel values.
    
    Parameters:
        mask (torch.Tensor or np.ndarray): The 2D mask (predicted or ground truth).
        title (str): Title for the histogram plot.
    """
    # Convert to NumPy if it's a Torch tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Flatten the mask to get pixel values
    mask_values = mask.flatten()

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(mask_values, bins=50, color='blue', alpha=0.7)
    plt.xlabel("Mask Pixel Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.show()


def get_model(num_keypoints=2,model_path=None, device="cuda",num_classes=2):
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

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")

    model.to(device).eval()

    return model


def draw_predictions(model, dataset, device, epoch, num_samples=8, result_dir='../result_dir'):
    """ Generate prediction visualization for selected samples. """
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    overlay_color = np.array([255, 0, 255], dtype=np.uint8)  # Green color
    alpha = 0.25  # Transparency factor (0 = fully transparent, 1 = solid color)
    
    for ax, idx in zip(axes.flatten(), indices):
        image, target = dataset[idx]

        img_np = np.array(F.to_pil_image(image))
        color_mask = np.zeros_like(img_np, dtype=np.uint8)

        # image = transform(image)  # Normalize image
        img_tensor = image.to(device).unsqueeze(0)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                output = model(img_tensor)[0]

        

        # Overlay masks and keypoints
        for i, mask in enumerate(output["masks"]):
            score = output["scores"][i].item()
            print(idx,i,score)
            if score < 0.5: 
                # print('Skipped')
                continue  # Skip low-confidence detections

            box = output['boxes'][i]
             # Draw bounding box
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow bbox


            # plot_mask_histogram(mask.cpu().numpy().squeeze(),title='raw')
            # plot_mask_histogram(torch.sigmoid(mask).cpu().numpy().squeeze(),title='sigmoid')


            mask = mask.cpu().numpy().squeeze() > 0.1
            # mask = torch.sigmoid(mask).cpu().numpy().squeeze() > 0.3
            # img_np[mask] = [0, 255, 0]  # Color overlay for masks
            color_mask[mask] = overlay_color

            # Draw keypoints
            if "keypoints" in output:
                keypoints = output["keypoints"][i].cpu().numpy()
                # print('keypoints',keypoints)
                for j, (x, y, v) in enumerate(keypoints):
                    if v > 0:  # Visibility flag
                        if j:
                            cv2.circle(img_np, (int(x), int(y)), 4, (0, 255, 0), -1)
                        else:
                            cv2.circle(img_np, (int(x), int(y)), 4, (255, 0, 0), -1)

        img_np = cv2.addWeighted(img_np, 1 - alpha, color_mask, alpha, 0)
        ax.imshow(img_np)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(result_dir + f"/predictions_epoch_{epoch}_offline.png")
    plt.close()
    print(f"Saved predictions to {result_dir}/predictions_epoch_{epoch}_offline.png")


def main():
    # Configurations
    data_root = "../dataset_generation_sps"
    model_path = "../results/maskrcnn_slip_0.pth"  # Update with the correct epoch/model path
    result_dir = "../results"
    num_samples = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = CocoDataset(json_path=data_root + "/short_slip_dataset_coco/val.json", img_dir=data_root + "")

    # Load trained model
    model = get_model(num_keypoints=2, model_path=model_path, device=device)

    # Run visualization
    draw_predictions(model, dataset, device, epoch=10, num_samples=num_samples, result_dir=result_dir)


if __name__ == "__main__":
    main()
