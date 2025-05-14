import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
from plot_metrics import MetricsPlotter  # Import our plotting module


from dataset import CocoDataset  
from visualize_predictions import draw_predictions
import torchvision
from tqdm import tqdm
import gc 
import random
import albumentations as A


# Load model
def get_model(num_keypoints=2,num_classes=2):
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

    return model

data_root = '../dataset_generation'
result_dir = '../results'


os.makedirs(result_dir, exist_ok=True)


# Initialize plotter
plotter = MetricsPlotter(result_dir=result_dir)

scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision
torch.backends.cudnn.benchmark = True


train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5),
    A.Normalize(mean=(0.486, 0.491, 0.488), std=(0.137, 0.129, 0.132)),
], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
   keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


# Training function
def train():
    train_dataset = CocoDataset(json_path=data_root+"/short_slip_dataset_coco/train.json", img_dir=data_root+"", transforms=train_transforms)
    val_dataset = CocoDataset(json_path=data_root+"/short_slip_dataset_coco/val.json", img_dir=data_root+"")  # Validation dataset

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2,collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_keypoints=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in tqdm(train_loader,desc='Epoch {0} processing'.format(epoch),unit='step'):
            # images = [img.to(device).half() for img in images]
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Print keypoints for debugging
            for i,t in enumerate(targets):
                # print("Keypoints:", t["keypoints"])
                keypoints = t['keypoints']
                num_instances = keypoints.shape[0]  # Number of objects in the image
                num_keypoints = keypoints.shape[1] // 3  # Number of keypoints per object
                t['keypoints'] = keypoints.view(num_instances, num_keypoints, 3)  # Reshape to (N, K, 3)
                targets[i] = t

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()
            epoch_loss += losses.item()

            # Print individual losses
            loss_details = {k: v.item() for k, v in loss_dict.items()}  # Convert tensors to floats
            print(f"Step Losses: {loss_details}")

        epoch_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        # Run evaluation
        ap, ar = evaluate(model, val_loader, data_root+"/short_slip_dataset_coco/val.json")
        draw_predictions(model, val_dataset, device, epoch+1, num_samples=8,result_dir=result_dir)
        
        # Update plots
        plotter.update(epoch+1, epoch_loss, ap, ar)
        plotter._plot()

        torch.save(model.state_dict(), result_dir+"/maskrcnn_slip_{0}.pth".format(epoch))
    print("Training complete!")
    # plotter.show_final()  # Keep plots open

# Evaluation function
def evaluate(model, data_loader, val_json_path, sample_ratio=0.05):
    print('Evaluating the model...')
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get 10% of the dataset indices
    all_indices = list(range(len(data_loader.dataset)))
    num_sampled_indices = int(sample_ratio * len(all_indices))  

    coco_gt = COCO(val_json_path)


    ap_sum = 0
    ar_sum = 0
    num_batches = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast():  # Enable mixed precision
            for j, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating", unit="batch")):
                if j >= num_sampled_indices:  # Process only 10% of the dataset
                    break
                # images = [img.to(device).half() for img in images]  # Convert to half precision
                images = [img.to(device) for img in images]
                outputs = model(images)

                results = []

                for i, output in enumerate(outputs):
                    image_id = targets[i]["image_id"].item()
                    for j in range(len(output["boxes"])):
                        box = output["boxes"][j].cpu().numpy().tolist()
                        score = output["scores"][j].item()
                        if score < 0.05:  # Filter low-confidence predictions
                            continue


                        label = int(output["labels"][j].item())

                        segmentation = None
                        if "masks" in output:
                            mask = output["masks"][j].cpu().numpy().squeeze(0)
                            segmentation = mask.tolist()

                        keypoints = None
                        if "keypoints" in output:
                            keypoints = output["keypoints"][j].cpu().numpy().flatten().tolist()

                        results.append({
                            "image_id": image_id,
                            "category_id": label,
                            "bbox": box,
                            "score": score,
                            "segmentation": segmentation,
                            "keypoints": keypoints
                        })

                 # Calculate COCO evaluation metrics for the current batch
                coco_dt = coco_gt.loadRes(results)
                coco_eval = COCOeval(coco_gt, coco_dt, "segm")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                # Update cumulative metrics
                ap_sum += coco_eval.stats[0]  # AP @ IoU=0.50
                ar_sum += coco_eval.stats[8]  # AR @ IoU=0.50
                num_batches += 1

                # Free GPU memory
                del images, outputs
                gc.collect()
                torch.cuda.empty_cache()

    # Calculate average AP and AR
    average_ap = ap_sum / num_batches if num_batches > 0 else 0
    average_ar = ar_sum / num_batches if num_batches > 0 else 0

    print(f"Average AP: {average_ap:.4f}, Average AR: {average_ar:.4f}")
    return average_ap, average_ar

if __name__ == "__main__":
    train()
