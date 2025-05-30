import numpy as np
import cv2
import time
import sys, os
from datetime import datetime
from math import *
import matplotlib.pyplot as plt

root = '../data_ours/validation_data_annotated/leaf/overlap'

sys.path.append('commons')
from cas_grasp_algo import run_grasp_algo
from utils_gs_cas import Parameters, create_directory

import sys
sys.path.append('../CNN_sps/')
from slip_maskrcnn_predictor import Predictor

model = Predictor('../final_weights/leaf_0.pth',num_classes=4)

#import file paths 

# data_path = root + '/ours_data'
data_path = root 
folder_path = data_path 
folder_path_2 = data_path 
out_final_path = data_path + '/grasp_pose'

# out_path = data_path
w = 640 #320
h = 480 #240
param = Parameters(w,h)
param.gripper_height = int(param.mh*70)  

inputs = {'param':param}
inputs['slanted_pose_detection'] = False #False
inputs['cone_detection'] = False #False
# self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
scan_names = sorted(list(set([os.path.basename(x)[0:3] 
	for x in os.listdir(folder_path)])))

#to pick only a selected range of files to run in code 
l = 1
m = 100
# scan_names = [str(i).zfill(6) for i in range(l, m)]

score_list = []



method = '/cas_method'
out_path = data_path+method
create_directory(out_final_path)

avg_time = 0.0
scan_names = ['1','2','3','4','5','6','7','8','9']
scenes = 1

for idx in scan_names:
	print('Processing the sample ',idx)
	if scenes == 101:
		break
	dmap = np.loadtxt(os.path.join(folder_path, idx)+'_depth_array.txt')
	img = cv2.imread(os.path.join(folder_path_2, idx)+'_image.jpg')
	# mask_path = f"{out_final_path}/masks_{idx}.npy"

	result_image, combine_masks, combine_keypoints = model.predict(img)
	# cv2.imwrite(out_final_path+'/{0}_prediction_target_slip.png'.format(idx),result_image)
	cv2.imwrite(out_final_path+'/{0}_predictions_all.png'.format(idx), result_image)




	l=l+1 #index of masks starts from zero 
	# mask = np.load(mask_path)

	# mask = create_a_single_mask(mask)
	#cv2.imwrite(out_path+method+'/final_image_{0}.png'.format(idx),final_image)
	
	# dt = datetime.now().strftime("%d%m%Y")
	# h,w,_ = img.shape
	# param = Parameters(w,h)
	

	inputs['image']= img
	inputs['darray'] = dmap
	inputs['depth_image'] = None
	inputs['final_attempt'] = True
	# inputs['dump_dir'] = out_path + '/' + idx
	inputs['dump_dir'] = None
	# inputs['median_depth_map'] = median_depth_map
	inputs['adaptive_clusters'] = False
	inputs['num_dirs'] = 6
	inputs['gqs_score'] = None

	inputs['keypoint'] = combine_keypoints
	inputs['mask'] = combine_masks
	# inputs['divide_masks'] = True 

	st = time.time()

	#************** main function calling here ********************
	result = run_grasp_algo(inputs)



	if scenes > 1:
		avg_time += (time.time()-st)
		# print('inference time',(time.time()-st))
		# print('avg_time',avg_time/(scenes-1))

	grasp = result[8]
	grasp_score = result[0][5]
	cluster_image = result[6]
	final_image = result[7]
	score_list.append(grasp_score)
	# bmap_vis = (bmap / bmap.max())*255
	# bmap_vis_denoised = (bmap_denoised / bmap_denoised.max())*255
	# cv2.imwrite(out_path+'/bmaps/bmap{0}.jpg'.format(idx),bmap_vis)#.astype(np.uint8))
	# cv2.imwrite(out_path+'/bmaps/bmap{0}_denoised.jpg'.format(idx),bmap_vis_denoised)#.astype(np.uint8))
	# if inputs['adaptive_clusters']:
	# 	method = '/baseline_adaptive'
	# else:
	# 	method = '/baseline'



	# path_final_pose = out_path + '/final_pose'
	# path_all_pose = out_path + '/final_pose'
	# path_segments = out_path + '/final_pose'
	# path_ = out_path + '/final_pose'
	# create_directory(path_final_pose)
	# create_directory(path_final_pose)
	# create_directory(path_final_pose)
	# create_directory(path_final_pose)

	cv2.imwrite(out_final_path+'/final_image_{0}.png'.format(idx),final_image)

	# np.savetxt(out_final_path+'/grasp_{0}.txt'.format(idx),grasp)
	# cv2.imwrite(out_final_path+'/cluster_{0}.png'.format(idx),cluster_image)
	# np.savetxt(out_final_path+'/score_list.txt',score_list,fmt='%s')
	# # print('avg_time',avg_time/scenes)
	# print('acc',np.count_nonzero(score_list)/scenes)
	scenes += 1
	# c = input('ruko. analyze karo.')

avg_time = avg_time/len(scan_names)
	# pc_cloud = np.loadtxt(os.path.join(data_path, idx)+'_pc.npy')
