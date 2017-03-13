import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle, cv2, math, glob
from scipy.ndimage.measurements import label
from feature_extraction import *
from Car import Car

# Load model and parameters
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
color_space = dist_pickle["color_space"]
print('Using:','orientations: ', orient, ', pix_per_cell: ', pix_per_cell, ', cell_per_block: ', cell_per_block, 
	', spatial_size: ', spatial_size, ', hist_bins: ', hist_bins, ', color_space: ', color_space)

history_len = 5
car = Car(history_len)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space):
	img = img.astype(np.float32)/255
	img_tosearch = img[ystart:ystop,:,:]
	ctrans_tosearch = convert_color(img_tosearch, color_space)

	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell)-1
	nyblocks = (ch1.shape[0] // pix_per_cell)-1 
	nfeat_per_block = orient*cell_per_block**2

	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell)-1 
	cells_per_step = 2  
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step

	# Compute individual channel HOG features for the entire image
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

	box_list = []
	all_windows = []
	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step

			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

			# Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

			# Scale features and make a prediction
			test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))	
			
			#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))	
			test_prediction = svc.predict(test_features)

			xbox_left = np.int(xleft*scale)
			ytop_draw = np.int(ytop*scale)
			win_draw = np.int(window*scale)
			all_windows.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))

			
			if test_prediction == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
			
				box = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart))	
				box_list.append(box)

	return box_list, all_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

#Pipeline function to find cars in each image frame
def pipeline(img):

	# Find cars use loaded model
	ystart = 400
	ystop = 656

	scale = 1.5
	box_list = []
	all_windows = []
	for y_start in [400, 464]:
		y_stop = y_start + 192
		#print(y_start, y_stop, scale)
		boxes, windows = find_cars(img, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
		box_list.extend(boxes)
		#print('windows:', windows)
		all_windows.append(windows)
		scale += 0.5		

	scale = 2.5
	boxes, windows = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
	box_list.extend(boxes)
	all_windows.append(windows)	

	#add detected boxed to car history
	car.add_car(box_list)
	
	# flatten list of list boxes
	boxes = sum(car.car_history, [])
	#print (boxes)

	# Use heat-map to remove false positives
	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	# Add heat to each box in box list
	heat = add_heat(heat, boxes)
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat, 4)
	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)
	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	heated_img = draw_labeled_bboxes(np.copy(img), labels)

	# Visualization
	# Draw car boxes
	box_img = np.copy(img)
	for box in box_list:
		cv2.rectangle(box_img,box[0], box[1], (0,0,255), 6) 
	
	# Draw windows
	windowed_img1 = np.copy(img)
	windowed_img2 = np.copy(img)
	windowed_img3 = np.copy(img)

	return heated_img


if __name__=="__main__":
	test_image_path = '/home/linfeng-zc/Documents/Udacity/CarND-Vehicle-Detection/test_images/'
	test_images = [file for file in glob.glob(test_image_path + '*.jpg', recursive=True)]
	for image in test_images:
		img = mpimg.imread(image)
		heated_img = pipeline(img)
