import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision.transforms import functional as F

def draw_predictions(model, dataset, device, epoch, num_samples=8,result_dir='../result_dir'):
	""" Generate prediction visualization for selected samples. """
	model.eval()
	fig, axes = plt.subplots(2, 4, figsize=(16, 8))
	
	indices = np.random.choice(len(dataset), num_samples, replace=False)
	
	for ax, idx in zip(axes.flatten(), indices):
		image, target = dataset[idx]

		img_np = np.array(F.to_pil_image(image))


		# image = transform(image)  # Normalize image
		img_tensor = image.to(device).unsqueeze(0)

		with torch.no_grad():
			with torch.cuda.amp.autocast(enabled=False):
				output = model(img_tensor)[0]

		

		# Overlay masks and keypoints
		for i, mask in enumerate(output["masks"]):
			score = output["scores"][i].item()
			# print(idx,i,score)
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
			img_np[mask] = [0, 255, 0]  # Color overlay for masks

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

		ax.imshow(img_np)
		ax.axis("off")

	plt.tight_layout()
	plt.savefig(result_dir+f"/predictions_epoch_{epoch}.png")
	plt.close()
