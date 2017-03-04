import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import json
from Line import Line
from fit_lines import *

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
	# Convert to grayscale
	if len(img.shape) == 3:
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	else:
		gray = img
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	# Return the result
	return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
	# Convert to grayscale
	if len(img.shape) == 3:
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	else:
		gray = img
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255 
	gradmag = (gradmag/scale_factor).astype(np.uint8) 
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	# Grayscale
	if len(img.shape) == 3:
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)	
	else:
		gray = img
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction, 
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output

# Define a function that thresholds the S-channel of HLS
def hls_select(img, channel, thresh=(0, 255)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	if channel == 's':
		single_channel = hls[:, :, 2]
	elif channel == 'h':
		single_channel = hls[:, :, 0]
	else:
		single_channel = hls[:, :, 1]

	binary_output = np.zeros_like(single_channel)
	binary_output[(single_channel > thresh[0]) & (single_channel <= thresh[1])] = 1
	return binary_output

# Define Region of Interest
def region_of_interest(img):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	imshape = img.shape[0:2]

	vertices=np.array([[(0,imshape[0]),(imshape[1]/2 - 2, imshape[0]* 0.59), (imshape[1]/2 + 2, imshape[0]* 0.59),(imshape[1], imshape[0])]], dtype=np.int32)

	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def pipline(img):
	
	masked_img = region_of_interest(img)
	
	s_img = hls_select(masked_img,'s', (100, 255))

	h_img = hls_select(masked_img,'h', (0, 80))
	
	hs_combined = np.zeros_like(s_img)
	hs_combined[((s_img == 1) & (h_img == 1))] = 1

	return hs_combined

def get_perspective_transform(source, destination):
	M = cv2.getPerspectiveTransform(source, destination)
	#warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	return M

def perspective_transform(img, perspective_matrix):
	img_size = (img.shape[1], img.shape[0])
	warped_img = cv2.warpPerspective(img, perspective_matrix, img_size, flags=cv2.INTER_LINEAR)
	return warped_img
	
def preprocess(image, mtx, dist):
	img = mpimg.imread(image)
	undist = cv2.undistort(img, mtx, dist)
	binary_img = pipline(undist)

if __name__=="__main__":
	test_images = glob.glob("/home/linfeng-zc/Documents/Udacity/CarND-Advanced-Lane-Lines/test_images/straight_lines1.jpg")

	# Load camera matrix and distort coeff
	mtx_list = []
	dist_list = []
	with open('camera_calibration.json') as camera_calibration:
		calib = json.load(camera_calibration)
		mtx_list = calib['camera_matrix']
		dist_list = calib['dist_coeff']

	mtx = np.array(mtx_list)
	dist = np.array(dist_list)	

	for image in test_images:
		img = mpimg.imread(image)
		undist = cv2.undistort(img, mtx, dist)
		binary_img = pipline(undist)
		perspective_matrix = get_perspective_transform(src, dst)
		binary_warped = perspective_transform(binary_img, perspective_matrix)
		undist_warped = perspective_transform(undist, perspective_matrix)
		left_line = Line()
		right_line = Line()
		left_fitx, right_fitx, ploty, curverad, offset = fit_line_splitter(binary_warped, left_line, right_line)

		x1 = int(np.average(left_fitx))
		x2 = int(np.average(right_fitx))
		y1 = int(np.amin(ploty))
		y2 = int(np.amax(ploty))
		print (x1, x2, y1, y2)
		
		cv2.rectangle(undist_warped, (x1, y1), (x2, y2), (255,0,0), 5)

		Minv = get_perspective_transform(dst, src)
		result = binary_warped = perspective_transform(undist_warped, Minv)

	
		f, axarr = plt.subplots(1, 3, figsize=(24, 9))
		f.tight_layout()
		axarr[0].imshow(img)
		axarr[1].imshow(undist_warped, cmap='gray')
		axarr[2].imshow(result, cmap='gray')
		
		plt.show()
		break
		
