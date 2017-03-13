import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel()
	# Return the feature vector
	return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	# Call with two outputs if vis==True
	if vis == True:
		features, hog_image = hog(img, orientations=orient, 
					  pixels_per_cell=(pix_per_cell, pix_per_cell),
					  cells_per_block=(cell_per_block, cell_per_block), 
					  transform_sqrt=False, 
					  visualise=vis, feature_vector=feature_vec)
		return features, hog_image

	# Otherwise call with one output
	else:	  
		features = hog(img, orientations=orient, 
			   pixels_per_cell=(pix_per_cell, pix_per_cell),
			   cells_per_block=(cell_per_block, cell_per_block), 
			   transform_sqrt=False, 
			   visualise=vis, feature_vector=feature_vec)
		return features

def convert_color(img, color_space):
	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else: 
		feature_image = np.copy(img)

	#if np.argmax(feature_image > 1):
	#	feature_image = feature_image.astype(np.float32)/255

	return feature_image	  

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
			hist_bins=32, orient=9, 
			pix_per_cell=8, cell_per_block=2, hog_channel=0,
			spatial_feat=True, hist_feat=True, hog_feat=True):	

	img_features = []

	feature_image = convert_color(img, color_space)
	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		img_features.append(spatial_features)
		#print("spatial_features: ", spatial_features.shape)

	if hist_feat == True:
		hist_features = color_hist(feature_image, nbins=hist_bins)
		img_features.append(hist_features)
		#print("color_features: ", hist_features.shape)

	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(get_hog_features(feature_image[:,:,channel], 
									orient, pix_per_cell, cell_per_block, 
									vis=False, feature_vec=True))	  
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)

		img_features.append(hog_features)

		#print("hog_features: ", hog_features.shape)

	return np.concatenate(img_features)

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
			hist_bins=32, orient=9,
			pix_per_cell=8, cell_per_block=2, hog_channel=0,
			spatial_feat=True, hist_feat=True, hog_feat=True):

	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		image = mpimg.imread(file)
		feature = single_img_features(image, color_space, spatial_size, hist_bins, orient,
					pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat) 
		
		features.append(feature)
	return features

if __name__=="__main__":
	filename = "/home/linfeng-zc/Documents/Udacity/CarND-Vehicle-Detection/vehicles/GTI_Far/image0000.png"
	image = mpimg.imread(filename);

	plt.imshow(image)
	features = single_img_features(image, color_space='HSV')
	print(features.shape)
