import numpy as np
import cv2




def binary_mask(arr,img_h=480,img_w=640):


	# Convert to integer type (required by OpenCV)
	arr = arr.astype(np.int32)

	# Create a blank binary mask (H=480, W=640)
	mask = np.zeros((img_h, img_w), dtype=np.uint8)

	# Fill the contour area with white (255) to create the instance mask
	cv2.fillPoly(mask, [arr], 255)

	# Display the mask
	# cv2.imshow("Instance Mask", mask)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# Save the binary mask (optional)
	cv2.imwrite("binary_mask.png", mask)

	return mask