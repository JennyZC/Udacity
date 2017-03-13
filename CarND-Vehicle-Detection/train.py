import numpy as np
import glob, random, time, pickle
import matplotlib.pyplot as plt
from feature_extraction import extract_features
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import matplotlib.image as mpimg

# train svm classifier
def train_svm_clf(car_file, notcar_file, test_size=0.2):
	color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9
	pix_per_cell = 8
	cell_per_block = 2
	hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
	spatial_size=(32, 32)
	hist_bins=32

	t=time.time()
	car_features = extract_features(car_file, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
					orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
					hog_channel=hog_channel)

	notcar_features = extract_features(notcar_file, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
					orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
					hog_channel=hog_channel)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to extract HOG features...')

	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)
	# Normalize features
	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=test_size, random_state=rand_state)

	print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))

	# Use a linear SVC 
	svc = LinearSVC()

	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')

	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

	# Check the prediction time for a single sample
	t=time.time()
	n_predict = 10
	print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
	print('For these',n_predict, 'labels: ', y_test[0:n_predict])
	t2 = time.time()
	print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
	
	svc_dict = {}
	svc_dict["svc"] = svc
	svc_dict["scaler"] = X_scaler
	svc_dict["orient"] = orient
	svc_dict["pix_per_cell"] = pix_per_cell
	svc_dict["cell_per_block"] = cell_per_block
	svc_dict["spatial_size"] = spatial_size
	svc_dict["hist_bins"] = hist_bins
	svc_dict["color_space"] = color_space

	return svc_dict

if __name__=="__main__":
	car_path = '/home/linfeng-zc/Documents/Udacity/CarND-Vehicle-Detection/vehicles/*/'
	notcar_path = '/home/linfeng-zc/Documents/Udacity/CarND-Vehicle-Detection/non-vehicles/*/'

	car_file = [file for file in glob.glob(car_path + '*.png', recursive=True)]
	notcar_file = [file for file in glob.glob(notcar_path + '*.png', recursive=True)]
	
	input_file_len = len(car_file) if len(car_file) < len(notcar_file) else len(notcar_file)

	random.shuffle(car_file)
	random.shuffle(notcar_file)

	print ('CAR FILE NUMBER: ' , len(car_file))
	print ('NOTCAR FILE NUMBER: ' , len(notcar_file))

	svc_dict = train_svm_clf(car_file, notcar_file)
	
	with open('svc_pickle.p', 'wb') as f:
		pickle.dump(svc_dict, f)

