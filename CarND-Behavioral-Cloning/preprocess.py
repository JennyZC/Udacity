import pickle, json, cv2
import numpy as np
import random
from scipy.misc import imread
from random import uniform
from sklearn.model_selection import train_test_split

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

data_file = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/track_data_new/driving_log.csv'
image_path = ''

ORG_ROW = 160
ORG_COL = 320 

ROI_ROW_START = 60 
ROI_ROW_END = 135
ROI_COL_START = 0
ROI_COL_END = ORG_COL

RESIZE_FACTOR = 5

ROWS = round((ROI_ROW_END - ROI_ROW_START) / RESIZE_FACTOR)
COLS = round((ROI_COL_END - ROI_COL_START) / RESIZE_FACTOR)
print("roi:", ROI_COL_START, ROI_COL_END, "row: ", ROWS, "Cols: ", COLS)
CHANNELS = 1

# Convert one channel 2D image to 3D image with dimention (rows, cols, 1)
def to_rank3(gray_image):
	result = np.zeros((gray_image.shape[0], gray_image.shape[1], 1), dtype=gray_image.dtype)
	result[:, :, 0] = gray_image
	return result

def flip_img(img):
	flipped_img = cv2.flip(img, 1)
	#return flipped_img
	return to_rank3(flipped_img)

# Preprocessing images
def preprocess(image):
	# To YUV
	img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

	# Crop ROI and take Y channel 
	img_roi = img_yuv[ROI_ROW_START:ROI_ROW_END, ROI_COL_START:ROI_COL_END, 0]
		
	# Resize image
	resized_roi = cv2.resize(img_roi, (COLS, ROWS))

	# Normalize image
	resized_roi = np.float32(resized_roi)
	resized_roi = (resized_roi - 128.0) / 128.0

	'''
	plt.figure(figsize=(1,1))
	plt.imshow(image)
	
	cv2.namedWindow("org", cv2.WINDOW_NORMAL)
	cv2.imshow("org", image)
	
	cv2.namedWindow("yuv", cv2.WINDOW_NORMAL)
	cv2.imshow("yuv", img_yuv)
	
	cv2.namedWindow("y", cv2.WINDOW_NORMAL)
	cv2.imshow("y", img_yuv[:, :, 0])

	cv2.namedWindow("resized_roi", cv2.WINDOW_NORMAL)
	cv2.imshow("resized_roi", resized_roi)
	cv2.waitKey(0)
	'''
	#return resized_roi
	return to_rank3(resized_roi)

def process_line(line, image_path):
	data = line.split(',')
	img_center = cv2.imread(image_path + data[0])#.astype(np.float32)

	img_left = cv2.imread(image_path + data[1])#.astype(np.float32)
	img_right = cv2.imread(image_path + data[2])#.astype(np.float32)
	correction = 0.02
	return [(img_center, float(data[3])), (img_left, float(data[3]) + correction), (img_right, float(data[3]) - correction)]

if __name__ == '__main__':
	pass
