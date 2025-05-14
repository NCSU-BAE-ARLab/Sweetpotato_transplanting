import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset import CocoDataset 


data_root = '../dataset_generation_sps'
dataset = CocoDataset(json_path=data_root+"/short_slip_dataset_coco/train.json", img_dir=data_root)
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

mean = torch.zeros(3)
std = torch.zeros(3)
n_samples = 0

for images, _ in loader:
    images = torch.stack(images) 
    images = images.view(images.shape[0], images.shape[1], -1)  # Flatten HxW
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_samples += images.shape[0]

mean /= n_samples
std /= n_samples
print(f"Mean: {mean.tolist()}, Std: {std.tolist()}")
