import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from color_gradient_threshold import preprocess
import glob
import json

def fit_line_splitter(binary_warped, left_line, right_line):
	last_n = 5
	if not left_line.detected or not right_line.detected:
		left_fit, right_fit = fit_line(binary_warped)
		left_line.detected = True
		right_line.detected = True
	else:
		left_fit, right_fit = fit_line_quick(binary_warped, left_line.current_fit, right_line.current_fit)
		if not left_fit or not right_fit:
			left_fit, right_fit = fit_line(binary_warped)

	if not left_fit or not right_fit:
		left_fit = left_line.best_fit
		right_fit = right_line.best_fit


	# Update current and previous coefficient difference
	left_line.diffs = np.absolute(left_fit - left_line.current_fit)
	right_line.diffs = np.absolute(right_fit - right_line.current_fit)

	# Update current coefficient
	left_line.current_fit = left_fit
	right_line.current_fit = right_fit

	# Update last n coefficient
	if len(left_line.recent_fit) == last_n:
		left_line.recent_fit.pop(0)
		right_line.recent_fit.pop(0)

	left_line.recent_fit.append(left_fit)
	right_line.recent_fit.append(right_fit)
	
	# Update best fit
	left_line.best_fit = np.average(np.asarray(left_line.recent_fit))
	right_line.best_fit = np.average(np.asarray(right_line.recent_fit))

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	return left_fitx, right_fitx, ploty

def fit_line(binary_warped):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:		
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	'''
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	'''
	return left_fit, right_fit

def fit_line_quick(binary_warped):
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	'''
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	'''

	return left_fit, right_fit

def warp_pespective(undist, binary_warped, left_fitx, right_fitx, ploty, Minv):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result



def window_mask(width, height, img_ref, center,level):
	output = np.zeros_like(img_ref)
	output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
	return output

def find_window_centroids(binary_warped, window_width, window_height, margin):

	window_centroids = [] # Store the (left,right) window centroid positions per level
	window = np.ones(window_width) # Create our window template that we will use for convolutions

	# First find the two starting positions for the left and right lane by using np.sum to 
	# get the vertical image slice
	# and then np.convolve the vertical image slice with the window template 

	# Sum quarter bottom of image to get slice, could use a different ratio
	l_sum = np.sum(binary_warped[int(3*binary_warped.shape[0]/4):,:int(binary_warped.shape[1]/2)], axis=0)
	l_center = np.argmax(np.convolve(window,l_sum, 'same'))#-window_width/2
	r_sum = np.sum(binary_warped[int(3*binary_warped.shape[0]/4):,int(binary_warped.shape[1]/2):], axis=0)
	r_center = np.argmax(np.convolve(window,r_sum, 'same'))+int(binary_warped.shape[1]/2)#-window_width/2+int(binary_warped.shape[1]/2)

	# Add what we found for the first layer
	window_centroids.append((l_center,r_center))

	# Go through each layer looking for max pixel locations
	for level in range(1,(int)(binary_warped.shape[0]/window_height)):
		# convolve the window into the vertical slice of the image
		image_layer = np.sum(binary_warped[int(binary_warped.shape[0]-(level+1)*window_height):
		int(binary_warped.shape[0]-level*window_height),:], axis=0)
		conv_signal = np.convolve(window, image_layer)
		# Find the best left centroid by using past left center as a reference
		# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
		offset = window_width/2
		l_min_index = int(max(l_center+offset-margin,0))
		l_max_index = int(min(l_center+offset+margin,binary_warped.shape[1]))
		l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
		# Find the best right centroid by using past right center as a reference
		r_min_index = int(max(r_center+offset-margin,0))
		r_max_index = int(min(r_center+offset+margin,binary_warped.shape[1]))
		r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
		# Add what we found for that layer
		window_centroids.append((l_center,r_center))

	return window_centroids

def show_window_centroids(binary_warped, window_centroids, window_width, window_height, margin):
	binary_warped[np.nonzero(binary_warped)] = 255
	if len(window_centroids) > 0:

		# Points used to draw all the left and right windows
		l_points = np.zeros_like(binary_warped)
		r_points = np.zeros_like(binary_warped)

		# Go through each level and draw the windows 	
		for level in range(0,len(window_centroids)):
			# Window_mask is a function to draw window areas
			l_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][0],level)
			r_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][1],level)
			# Add graphic points from window mask here to total pixels found 
			l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
			r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

		# Draw the results
		template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
		zero_channel = np.zeros_like(template) # create a zero color channle 
		template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
		warpage = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8) # making the original road pixels 3 color channels
	
		output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

	# If no window centers found, just display orginal road image
	else:
		output = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8)

	# Display the final results
	plt.imshow(output)
	plt.title('window fitting results')
	plt.show()

if __name__=="__main__":
	test_images = glob.glob("/home/linfeng-zc/Documents/Udacity/CarND-Advanced-Lane-Lines/test_images/*")

	# Load camera matrix and distort coeff
	mtx_list = []
	dist_list = []
	with open('camera_calibration.json') as camera_calibration:
		calib = json.load(camera_calibration)
		mtx_list = calib['camera_matrix']
		dist_list = calib['dist_coeff']

	mtx = np.array(mtx_list)
	dist = np.array(dist_list)

	window_width = 50 
	window_height = 80 # Break image into 9 vertical layers since image height is 720
	margin = 100 # How much to slide left and right for searching	

	for image in test_images:
		binary_warped = preprocess(image, mtx, dist)

		window_centroids = find_window_centroids(binary_warped, window_width, window_height, margin)
		show_window_centroids(binary_warped, window_centroids, window_width, window_height, margin)

		'''
		fit_line(binary_warped)
			
		f, axarr = plt.subplots(1, 2, figsize=(24, 9))
		f.tight_layout()
		axarr[0].imshow(binary_warped, cmap='gray')
		'''
