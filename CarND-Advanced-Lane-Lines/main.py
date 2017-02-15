import os.path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import glob
import json
import cv2
import numpy as np
from camera_calibration import *
from color_gradient_threshold import *
from fit_lines import *

def calibrate(calib_image_path, calib_json, camera_matrix, dist_coeff):
	# check if camera parameters exists
	if os.path.isfile(calib_json):
		with open(calib_json) as camera_calibration:
			calib = json.load(camera_calibration)
			mtx = np.array(calib[camera_matrix])
			dist = np.array(calib[dist_coeff])
	else:
		# Chessboard shape
		nx = 9
		ny = 6
		# Calibrate images
		calib_images = glob.glob(calib_image_path)
		ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images, (nx, ny))

		if (ret):
			data = {camera_matrix: mtx.tolist(), dist_coeff: dist.tolist()}
			with open(calib_json, "w") as f:
				json.dump(data, f)

	return mtx, dist

def process_image(img):

	# Calibrate camera
	calib_image_path = "/home/linfeng-zc/Documents/Udacity/CarND-Advanced-Lane-Lines/camera_cal/*"
	calib_json = 'camera_clibration.json'
	camera_matrix = 'camera_matrix'
	dist_coeff = 'dist_coeff'
	mtx, dist = calibrate(calib_image_path, calib_json, camera_matrix, dist_coeff)

	# Get perspective transform matrix
	src = np.float32(
		[[511, 511],
		 [781, 511],
		 [438, 564],
		 [859, 564]])

	dst = np.float32(
		[[465, 520],
		 [815, 520],
		 [465, 620],
		 [815, 620]])

	M = get_perspective_transform(src, dst)
	Minv = get_perspective_transform(dst, src)

	# Get binary warped image
	undist = cv2.undistort(img, mtx, dist)
	binary_img = pipline(undist)
	binary_warped = perspective_transform(binary_img, M)

	# Fit line
   	global left_line, right_line
    	left_fitx, right_fitx, ploty = fit_line_splitter(binary_warped, left_line, right_line)

    	result = warp_pespective(undist, binary_warped, left_fitx, right_fitx, ploty, Minv)

    	return result


# Get lane binary image with color/gradient
'''
sample_image_path = ('/home/linfeng-zc/Documents/Udacity/CarND-Advanced-Lane-Lines/test_images/test6.jpg')
img = mpimg.imread(sample_image_path)
result = process_image(img)
f, axarr = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
axarr[0].imshow(result)
plt.show()
'''
left_line = Line()
right_line = Line()
white_output = 'project_video_result.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False
