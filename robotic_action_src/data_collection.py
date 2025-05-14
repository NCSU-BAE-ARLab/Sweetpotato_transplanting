#!/usr/bin/env python3

import numpy as np
import cv2
from camera import Camera

def collect_n_save_data(path,cam):

	for i in range(0,10):
	
		input('Make sure your scene is ready. Then press and enter any key (e.g. 1)')
		cam.click_a_depth_sample()
		np.savetxt(path+'/{0}_depth_array.txt'.format(i),cam.cur_depth_map)

		cam.click_an_image_sample()
		cv2.imwrite(path+'/{0}_image.jpg'.format(i),cam.cur_image)

		print('Data captured!',i)

	print('done!!')

if __name__ == '__main__':
	path='../data_ours'
	cam = Camera('temp')
	collect_n_save_data(path,cam)