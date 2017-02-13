import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import glob
import json


def show_imgs(calib_images, mtx, dist, chess_board_shape):
	for calib_image in calib_images:
		img = cv2.imread(calib_image)

		# Find chessboard corners
		ret, corners = cv2.findChessboardCorners(img, chess_board_shape, None)
		
		# undistort image
		undist = cv2.undistort(img, mtx, dist)

		print("Find corner: ", ret)

		# Draw and display the corners
		cv2.drawChessboardCorners(img, chess_board_shape, corners, ret)
		cv2.imshow("original", img)

		cv2.imshow("undist", undist)
		cv2.waitKey(0)
	
# Function used to calibrate camera
# @param img_files A list of image names with global path
# @param chess_board_shape Shape of the chess borad (nx, ny)
def calibrate_camera(img_files, chess_board_shape):
	# Arrays to store object points and image points from all the images
	obj_points = []
	img_points = []

	# Prepare object points (0, 0, 0), (1, 0, 0), (2, 0, 0) ..., (9, 6, 0)
	obj_p = np.zeros((chess_board_shape[0] * chess_board_shape[1], 3), np.float32)
	obj_p[:, :2] = np.mgrid[0 : chess_board_shape[0], 0 : chess_board_shape[1]].T.reshape(-1, 2)

	for calib_image in img_files:
		# Read image
		img = cv2.imread(calib_image)

		# Convert to gray scale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find chessboard corners
		ret, corners = cv2.findChessboardCorners(img, chess_board_shape, None)

		if ret == True:
			obj_points.append(obj_p)
			img_points.append(corners)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

	total_error = 0.
	for i in range(len(obj_points)):
		img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv2.norm(img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)
		total_error += error

	print ("mean reprojection error: ", total_error/len(obj_points))

	return ret, mtx, dist, rvecs, tvecs

if __name__=="__main__":
	# Chessboard shape
	nx = 9
	ny = 6

	# Calibrate images
	calib_images = glob.glob("/home/linfeng-zc/Documents/Udacity/CarND-Advanced-Lane-Lines/camera_cal/*")

	ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images, (nx, ny))
	#show_imgs(calib_images, mtx, dist, (nx, ny))
	
	print(mtx, dist)

	if (ret):
		data = {"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist()}
		fname = "camera_calibration.json"
		with open(fname, "w") as f:
			json.dump(data, f)

