import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def major_minor_axis(image):
    points = np.argwhere(image)
    X = points[:,0]
    Y = points[:,1]
    x = X - np.mean(X)
    y = Y - np.mean(Y)
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]] 
    x_v2, y_v2 = evecs[:, sort_indices[1]]
    theta = -np.arctan((y_v1)/(x_v1))
    
    rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
					  [np.sin(theta), np.cos(theta)]])
    transformed_mat = rotation_mat * np.vstack([X, Y])
    x_transformed, y_transformed = transformed_mat.A
    major = np.max(x_transformed)-np.min(x_transformed)
    minor = np.max(y_transformed)-np.min(y_transformed)
    return theta,major,minor

def axis_angle(image):
    points = np.argwhere(image)
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    modi_x = np.array(points[:, 0] - cx)
    modi_y = np.array(points[:, 1] -  cy)
    num = np.sum(modi_x * modi_y)
    den = np.sum(modi_x ** 2 - modi_y ** 2)
    angle = 0.5 * math.atan2(2 * num, den)
    axis_dict = {
		"angle": angle,
		"centroid": (cx, cy)}
    return axis_dict

def calculate_center(mask):
    indices = np.argwhere(mask)  # Gets the indices of all true values in the mask
    center = np.mean(indices, axis=0) 
    return center

def rotate_boolean_array(array, angle, center):
    angle_deg = np.degrees(angle)
    # print(angle_deg)
    rows, cols = array.shape
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1)
    rotated_array = cv2.warpAffine(array.astype(np.uint8), rotation_matrix, (cols, rows))
    rotated_array = (rotated_array != 0)

    return rotated_array

def crop_binary_mask(image, width):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(image.astype(np.uint8))
    cluster_index = 1  
    selected_cluster = np.where(labels == cluster_index, 1, 0)
    kernel = np.ones((5, 5), np.uint8)
    dilated_cluster = cv2.dilate(selected_cluster.astype(np.uint8), kernel, iterations=1)

    split_point = int(width[0]) 
    split_mask_1 = dilated_cluster.copy()
    split_mask_1[:, split_point:] = 0
    split_mask_2 = dilated_cluster.copy()
    split_mask_2[:, :split_point] = 0

    
    cropped_image_1 = np.where(split_mask_1 == 1, image, 0)
    cropped_image_2 = np.where(split_mask_2 == 1, image, 0)

    cropped_image_1[np.logical_and(cropped_image_1 != 1, selected_cluster != 1)] = 0
    cropped_image_2[np.logical_and(cropped_image_2 != 1, selected_cluster != 1)] = 0

    return cropped_image_1, cropped_image_2
