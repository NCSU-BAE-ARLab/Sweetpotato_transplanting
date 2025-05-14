import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
import numpy as np
import cv2
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from skimage.draw import polygon


# Custom dataset class for COCO format (same as before)
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_dir, transforms=None):
        self.transforms = transforms
        self.img_dir = img_dir

        # Load COCO annotations
        with open(json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())

        self.height = self.coco_data["images"][0]["height"]
        self.width = self.coco_data["images"][0]["width"]

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        
        # Load image
        # print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize

        # Get annotations
        annos = self.annotations[img_id]

        boxes, labels, masks, keypoints = [], [], [], []
        for ann in annos:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # Convert to (x1, y1, x2, y2)
            # print(ann['category_id'])
            labels.append(ann['category_id'])  # "slip" class (background = 0)
            masks.append(self.decode_rle(ann["segmentation"]))  # Convert RLE mask to binary
            keypoints.append(ann["keypoints"])


        # keypoints_np = np.array(keypoints, dtype=np.float32)

        # # Ensure keypoints are always in (N, 2, 3) format
        # if keypoints_np.ndim == 2:  # If (2, 3), add a batch dimension
        #     keypoints = keypoints_np[np.newaxis, :, :]  # Becomes (1, 2, 3)
        # elif keypoints_np.ndim == 1:  # If empty, reshape properly
        #     keypoints = keypoints_np.reshape(0, 2, 3)
        # else:
        #     keypoints = keypoints_np


        processed_keypoints = []
        for kp in keypoints:
            if kp:  # If keypoints exist for this instance
                processed_keypoints.append(kp)
            else:  # If no keypoints, append an empty list
                processed_keypoints.append([0,0,0,0,0,0])

        # Convert keypoints, ensuring it's an array of shape (N, ?, 3)
        keypoints = np.array(processed_keypoints) 

        if self.transforms:
            # print(img.shape)
            # print(np.array(keypoints)[:, :, :2].shape)
            # print('before',np.array(keypoints))

             


            # bboxes=np.array(boxes)
            # masks=np.array(masks)
            # print(keypoints)
            # keypoints=np.array(keypoints)
            # labels=np.array(labels)
            transformed = self.transforms(image=img, bboxes=np.array(boxes), masks=np.array(masks), keypoints=keypoints,labels=np.array(labels)) #[:, :, :2]
            # print('after',transformed["keypoints"])
            image = transformed["image"]
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize
            target = {
                "boxes": torch.tensor(np.array(transformed["bboxes"]), dtype=torch.float32),
                "labels": torch.tensor(np.array(labels), dtype=torch.int64),
                "masks": torch.tensor(np.array(transformed["masks"]), dtype=torch.uint8),
                "keypoints": torch.tensor(np.array(transformed["keypoints"]), dtype=torch.float32),
                # "keypoints_visible": torch.tensor([2] * len(keypoints), dtype=torch.uint8)
                "keypoints_visible":torch.tensor(np.array([kp[2] for kp in transformed["keypoints"]]), dtype=torch.uint8),
                "image_id": torch.tensor(np.array([img_id]), dtype=torch.int64)
            }
            

        else:
            image = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize
            # Convert to tensors
            target = {
                "boxes": torch.tensor(np.array(boxes), dtype=torch.float32),
                "labels": torch.tensor(np.array(labels), dtype=torch.int64),
                "masks": torch.tensor(np.array(masks), dtype=torch.uint8),
                "keypoints": torch.tensor(np.array(keypoints), dtype=torch.float32),
                # "keypoints_visible": torch.tensor([2] * len(keypoints), dtype=torch.uint8)
                "keypoints_visible":torch.tensor(np.array([kp[2] for kp in keypoints]), dtype=torch.uint8),
                "image_id": torch.tensor(np.array([img_id]), dtype=torch.int64)
            }

        return image, target

    def __len__(self):
        return len(self.image_ids)

    def decode_rle(self, rle):
        """ Convert COCO RLE to binary mask. """
        if isinstance(rle, list):
            # Convert polygon format to mask (Not RLE)
            # print('Not RLE')
            mask =  np.zeros((self.height,self.width), dtype=np.uint8)
            for polygon_pts in rle:  # Multiple polygons per object
                # Convert polygon list to (r, c) coordinates
                x = polygon_pts[0::2]  # x coordinates
                y = polygon_pts[1::2]  # y coordinates
                rr, cc = polygon(y, x, shape=(self.height, self.width))

                # Fill the mask
                mask[rr, cc] = 1

            return mask
        else:
            # Convert RLE to binary mask
            print('RLE')
            from pycocotools import mask as maskUtils
            return maskUtils.decode(rle)

