import cv2
import numpy as np
import json
import sys

if __name__=="__main__":
	image_path = "/home/linfeng-zc/Documents/Udacity/CarND-Advanced-Lane-Lines/test_images/"
	undist_path = "/home/linfeng-zc/Documents/Udacity/CarND-Advanced-Lane-Lines/undist_images/"
	if len(sys.argv) < 2:
		image_name = "straight_lines2.jpg"	
	else:
		image_name = str(sys.argv[1])

	# Load camera matrix and distort coeff
	mtx_list = []
	dist_list = []
	with open('camera_calibration.json') as camera_calibration:
		calib = json.load(camera_calibration)
		mtx_list = calib['camera_matrix']
		dist_list = calib['dist_coeff']

	mtx = np.array(mtx_list)
	dist = np.array(dist_list)	

	print(image_path + image_name)
	img = cv2.imread(image_path + image_name)

	undist_name = image_name[:-4] + "_undist.jpg";
	undist = cv2.undistort(img, mtx, dist)
	cv2.imwrite(undist_path + undist_name, undist)
	
