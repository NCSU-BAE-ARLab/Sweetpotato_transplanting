import numpy as np
import sys
import cv2
import copy
import time
from utils_gs_cas import draw_samples
from utils_gs_cas import create_directory
from utils_gs_cas import final_axis_angle
from utils_gs_cas import draw_rectified_rect
from grasp_evaluation import calculate_GDI2
from custom_grasp_planning_algorithm_dense_cas import select_a_best_grasp_pose
from utils_gs_cas import remove_outlier
from utils_gs_cas import divide_object_masks_along_their_longer_axis


def get_other_point_from_mask(mask, point):
    """
    Args:
        mask: 2D binary numpy array
        point: tuple (x, y) — reference point

    Returns:
        high_end: np.array([x, y]) — farthest end from point
        low_end: np.array([x, y]) — closest end to point
    """
    # Step 1: Get coordinates of the foreground pixels
    ys, xs = np.nonzero(mask)
    coords = np.vstack((xs, ys)).T  # Shape: (N, 2)

    # Step 2: Apply PCA to find major axis
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]  # Unit vector

    # Step 3: Project points onto the major axis
    projections = centered @ major_axis
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)

    # Step 4: Get the two endpoints
    pt1 = coords[max_idx]  # first endpoint
    pt2 = coords[min_idx]  # second endpoint

    # Step 5: Compare distance to input point
    point = np.array(point)
    d1 = np.linalg.norm(pt1 - point)
    d2 = np.linalg.norm(pt2 - point)

    if d1 > d2:
        return pt1
    else:
        return pt2


def get_direction_from_keypoints(keypoints, mask):
    direction = 0 # default direction
    if keypoints['head'] is None:
        if keypoints['tail'] is None:
            return direction
        else:
            tail = keypoints['tail']
            head = get_other_point_from_mask(mask, tail)
    else:
        if keypoints['tail'] is None:
            head = keypoints['head']
            tail = get_other_point_from_mask(mask, head)
        else:
            head = keypoints['head']
            tail = keypoints['tail']
    if head[1] < tail[1]:
        direction = 1
    return direction



def calculate_center(mask):
    indices = np.argwhere(mask)  # Get the indices of all true values in the mask
    center = np.mean(indices, axis=0) 
    return center


def remove_outliers_cluster_wise(points,labels):
    ids = np.unique(labels)
    output_points = []
    output_labels = []
    for i in ids:
        cluster_mask = (labels==i)
        cluster = points[cluster_mask]
        cluster_labels = labels[cluster_mask]
        cluster, filter_mask = remove_outlier(cluster)
        output_points.extend(cluster.tolist())
        output_labels.extend(cluster_labels[filter_mask].tolist())
    return np.array(output_points), np.array(output_labels)

def get_seg_mask(mask):
    n,w,h = mask.shape
    seg_mask = np.zeros((w,h))
    for i in range(n):
        seg_mask = np.where(mask[i],i+1,seg_mask)
    return seg_mask

def run_grasp_algo(inputs):
    img = inputs['image']
    darray = inputs['darray']
    depth_image = inputs['depth_image']
    param = inputs['param'] 
    final_attempt = inputs['final_attempt']
    path = inputs['dump_dir']
    # median_depth_map = inputs['median_depth_map']
    
    all_mask = inputs['mask']
    all_keypoints = inputs['keypoint']

    inputs['seg_mask'] = get_seg_mask(all_mask)

    if path is not None:
        create_directory(path+'/bmaps')
        create_directory(path+'/directions')
        create_directory(path+'/grasp_pose_info')

    new_img = copy.deepcopy(img)
    clustter_img= copy.deepcopy(img)
    final_pose_rect_img = copy.deepcopy(img)
    depth_image_copy = copy.deepcopy(depth_image)
    img_copy = copy.deepcopy(img)
    initial_img = copy.deepcopy(img)

    # mask = mask[0] # retrive only top grasp pose


    for j in range(all_mask.shape[0]):
        print('Iteration:',j+1)
        selected_masks = [all_mask[j]]  # select the slips in order of their confidence scores
    #*************** NEW CODE FOR CENTER ******************************

        center_list=[]
        for mask in selected_masks:
            center = calculate_center(mask)
            center_list.append(center)
        centers = np.array(center_list)
        # print('centers', centers)
        angles = []
        for k in range(len(centers)):
            indices = np.argwhere(selected_masks [k])
            dict = param.axis_angle(points=indices)
            minor_points = dict["minor_axis_points"]
            angle = dict["angle"]
            points = indices
            angles.append(angle)
        centers = centers[:, ::-1]
        centers = np.array(centers)
        # print('centers', centers)
        angles = np.array(angles)
        inputs['top_grasp_points'] = centers
        inputs['angles'] = angles

    #****************** the main function *******************     
        grasp_pose_info = select_a_best_grasp_pose(inputs)
        
        final_rect_pixel_array = grasp_pose_info['final_pose_rectangle']
        gdi2 = grasp_pose_info['gdi_calculator']
        original_idx = grasp_pose_info['selected_idx']
        gdi_old_way = grasp_pose_info['gdi_old_way']
        keypoints = all_keypoints[j]
        mask = all_mask[j] 

        if not gdi_old_way:
            print('Found solution in Iteration:',j+1)
            break


    grasp_direction = get_direction_from_keypoints(keypoints, mask)
    print('************ grasp_direction *************', grasp_direction)
    angle = final_axis_angle(final_rect_pixel_array)
    
    grasp_score = 0.0
    valid_flag = False
    if not gdi_old_way:
        valid_flag = True
        # print('original_idx',original_idx)
        # print('final pose',final_pose_rect_img)
        final_pose_rect_img = 0.8*final_pose_rect_img
        new_centroid, new_gripper_opening, object_width = gdi2.draw_refined_pose(final_pose_rect_img, thickness=4)
        # if path is not None:
        gdi2.draw_refined_pose(depth_image_copy)
        # cv2.imwrite(dump_dir+'/depth_image.png',depth_image_copy)
        grasp_score = (gdi2.FLS_score + gdi2.CRS_score)/2
        print('-------------------------------------------------')
        print(gdi2.FLS_score  ,       gdi2.CRS_score )

        cx = new_centroid[0]
        cy = new_centroid[1]
        gripper_opening = (float(new_gripper_opening)/param.gripper_finger_space_max)*param.Max_Gripper_Opening_value
        if gripper_opening > 1.0:
            gripper_opening = 1.0

    else:
        cx = final_rect_pixel_array[0][0]
        cy = final_rect_pixel_array[0][1]
        gripper_opening = 1.0
        grasp_score = (gdi2.FLS_score)
        # if path is not None:
        draw_rectified_rect(img=final_pose_rect_img, pixel_points=final_rect_pixel_array)

    min_depth_difference = gdi2.min_depth_difference
    print('gdi2.min_depth_difference',gdi2.min_depth_difference)

    if path is not None:
        cv2.imwrite(path+'/{0}_final.jpg'.format(inputs['exp_num']), final_pose_rect_img)
        # cv2.imwrite(path+'/bmap.jpg',grasp_pose_info['bmap'])
        # cv2.imwrite(path+'/bmap_ws.jpg',grasp_pose_info['bmap_ws'])
        np.savetxt(path+'/their_idx.txt',[original_idx])
        # np.savetxt(path+'/gdi.txt',np.array(gdi2.GDI))
        # np.savetxt(path+'/gdi_plus.txt',np.array(gdi2.GDI_plus))
    # cv2.imwrite(path+'/top_3_recs.jpg', depth_image)
    # cv2.imwrite(path+'/all_poses_depth.jpg', depth_image_copy)
    # cv2.imwrite(path+'/all_poses_rgb.jpg', img_copy)

    # print('cx cy', cx, cy)
    z = darray[int(cy)][int(cx)]
    x,y = param.pixel_to_xyz(cx,cy,z)
    # print('FOV', x,y,z)
    fov = np.array([x,y,z])
    # st = time.time()
    # cx = (3.2/mw)*cx
    # cy = (2.4/mh)*cy
    # x1,y1,z1 = query_point_cloud_client(cx, cy)
    # # try:
    # #     x,y,z = get_3d_cam_point(np.array([cx, cy])).cam_point
    # # except rospy.ServiceException as e:
    # #     print("Point cloud Service call failed: %s"%e)
    # # print('time searching',time.time()-st)
    # print('PCD', x1,y1,z1)
    target = [x,y,z,angle,gripper_opening,grasp_score,grasp_direction]#,new_gripper_opening]
    grasp = [cx,cy,angle,gripper_opening,valid_flag,grasp_direction]
    print('target',target)
    if path is not None:
        np.savetxt(path+'/target.txt',target,fmt='%f') 
        np.savetxt(path+'/center.txt',np.array([cx,cy]),fmt='%d')
    # np.savetxt(path+'/manualseed.txt',np.array([manualSeed]),fmt='%d')
    
    boundary_pose = False
    # min_depth_difference = 0.02

    # outputs = {'grasp_score':grasp_score}

    return target,True,np.array([cy,cx]),valid_flag,boundary_pose, min_depth_difference,clustter_img, final_pose_rect_img, grasp


