import json, cv2
import numpy as np
from scipy.misc import imread
from random import uniform
from keras.layers import Input, Activation, Dropout, Dense, Flatten, Lambda
from keras.models import Sequential, model_from_json, load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from preprocess import *
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


NUM_EPOCHS = 10
BATCH_SIZE = 8

def load_samples(file_name):
	samples = []
	with open(file_name) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
	return samples

def get_generator(samples, batch_size=32):
	shuffle(samples)
	sample_size = len(samples)
	while 1:
		X_batch = []
		y_batch = []
		for start_i in range(0, sample_size, batch_size):
			end_i = start_i + batch_size 
			for line in samples[start_i:end_i]:
				image_tuples = process_line(line)
				for image_tuple in image_tuples:
					preprocessed_image = preprocess(image_tuple[0])				
					X_batch.append(preprocessed_image)
					y_batch.append(image_tuple[1])
					
					'''
					flipped = flip_img(preprocessed)
					X_batch.append(flipped)
					y_batch.append(-y)
					'''
					X_train = np.array(X_batch)
					y_train = np.array(y_batch)
			yield shuffle(X_train, y_train)

def get_nvidia_model():

	kernel_size = (3, 3)
	
	model = Sequential()

	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(ROWS, COLS, CHANNELS)))
	
	'''
	24, 36, 48, 64, 64
	'''
	model.add(Convolution2D(24, kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))

	model.add(Convolution2D(36, kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))

	model.add(Convolution2D(48, kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))

	model.add(Convolution2D(64, kernel_size[0], kernel_size[1],  border_mode='valid', activation='relu'))

	model.add(Convolution2D(64, kernel_size[0], kernel_size[1],  border_mode='valid', activation='relu'))


	model.add(Flatten())

	'''
	1164, 500, 100, 50, 10, 1
	'''
	model.add(Dense(1164, activation='relu'))

	model.add(Dense(500, activation='relu'))

	model.add(Dense(100, activation='relu'))

	model.add(Dense(50, activation='relu'))

	model.add(Dense(10, activation='relu'))

	model.add(Dense(1, activation='linear'))

	model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
	return model

def get_model():
	'''
	lr = 0.0001
	weight_init='glorot_normal'
	opt = RMSprop(lr)
	loss = 'mean_squared_error'

	model = Sequential()

	model.add(BatchNormalization(mode=2, axis=1, input_shape=(ROWS, COLS, CHANNELS)))
	model.add(Convolution2D(3, 3, 3, init=weight_init, border_mode='valid', activation='relu', input_shape=(ROWS, COLS, CHANNELS)))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(9, 3, 3, init=weight_init, border_mode='valid', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(18, 3, 3, init=weight_init, border_mode='valid', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(32, 3, 3, init=weight_init, border_mode='valid',  activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(80, activation='relu', init=weight_init))

	model.add(Dense(15, activation='relu', init=weight_init))

#	model.add(Dropout(0.25))
	model.add(Dense(1, init=weight_init, activation='linear'))

	model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
	'''	

	# number of convolutional filters to use
	nb_filters1 = 16
	nb_filters2 = 8
	nb_filters3 = 4
	nb_filters4 = 2

	# size of pooling area for max pooling
	pool_size = (2, 2)

	# convolution kernel size
	kernel_size = (3, 3)

	# Initiating the model
	model = Sequential()

	# Starting with the convolutional layer
	# The first layer will turn 1 channel into 16 channels
	model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
				border_mode='valid',
				input_shape=(ROWS, COLS, CHANNELS)))
	# Applying ReLU
	model.add(Activation('relu'))
	# The second conv layer will convert 16 channels into 8 channels
	model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
	# Applying ReLU
	model.add(Activation('relu'))
	# The second conv layer will convert 8 channels into 4 channels
	model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))
	# Applying ReLU
	model.add(Activation('relu'))
	# The second conv layer will convert 4 channels into 2 channels
	model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))
	# Applying ReLU
	model.add(Activation('relu'))
	# Apply Max Pooling for each 2 x 2 pixels
	model.add(MaxPooling2D(pool_size=pool_size))
	# Apply dropout of 25%
	model.add(Dropout(0.25))

	# Flatten the matrix. The input has size of 360
	model.add(Flatten())
	# Input 360 Output 16
	model.add(Dense(16))
	# Applying ReLU
	model.add(Activation('relu'))
	# Input 16 Output 16
	model.add(Dense(16))
	# Applying ReLU
	model.add(Activation('relu'))
	# Input 16 Output 16
	model.add(Dense(16))
	# Applying ReLU
	model.add(Activation('relu'))
	# Apply dropout of 50%
	model.add(Dropout(0.5))
	# Input 16 Output 1
	model.add(Dense(1))

	model.compile(loss='mean_squared_error',
	      optimizer=Adam(),
	      metrics=['accuracy'])

	return model

#TODO
def three_imgs_test():
	data_file = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/example_data/TEST_IMG/driving_log.csv'
	image_path = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/example_data/'

	img1 = imread(image_path + 'IMG/center_2016_12_01_13_34_06_150.jpg')
	preprocessed_img1 = preprocess(img1)

	#img2 = imread(image_path + 'IMG/center_2016_12_01_13_35_19_746.jpg')
	img2 = imread('/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/track_data_ii/IMG/center_2017_02_04_20_07_35_350.jpg')
	preprocessed_img2 = preprocess(img2)

	img3 = imread(image_path + 'IMG/center_2016_12_01_13_34_01_377.jpg')
	preprocessed_img3 = preprocess(img3)

	#model.fit(preprocessed_img1[None, :, :, :], np.array([0.0617599]), nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2)

	cv2.namedWindow("img", cv2.WINDOW_NORMAL)
	cv2.imshow("img", img2) 

	cv2.namedWindow("preprocessed_img", cv2.WINDOW_NORMAL)
	cv2.imshow("preprocessed_img", preprocessed_img2[ :, :, 0])
	cv2.waitKey(0)

	steering_angle1 = float(model.predict(preprocessed_img1[None, :, :, :], batch_size=1))
	steering_angle2 = float(model.predict(preprocessed_img2[None, :, :, :], batch_size=1))

	print ('steering angle: ', steering_angle1, steering_angle2)

if __name__ == '__main__':
	file_name = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/track_data_new/driving_log.csv'
	samples = load_samples(file_name)
	
	train_samples, test_samples = train_test_split(samples, test_size=0.1)
	train_samples, validation_samples = train_test_split(train_samples, test_size=0.2)
	
	batch_size = 8

	#test generator
	train_generator = get_generator(train_samples, batch_size)
	validation_generator = get_generator(validation_samples, batch_size)
	
	'''
	x, y = next(generator)
	print("x: ", x.shape)
	print("y: ", y)

	for image in x:
		print(image.shape)
		cv2.namedWindow("test_generator", cv2.WINDOW_NORMAL)
		cv2.imshow("test_generator", image[:, :, 0])
		cv2.waitKey(0)
	'''
	model = get_nvidia_model()
	#model = get_model()
	model.summary()

	model.fit_generator(train_generator, nb_epoch=NUM_EPOCHS, samples_per_epoch=len(train_samples), 
		validation_data=validation_generator, nb_val_samples=len(validation_samples), max_q_size=5)

	model.save('saved_model/model.h5')
	print("Saved model to disk")

	#model.fit_generator(get_generator(train_lines, BATCH_SIZE), 
	#	nb_epoch=NUM_EPOCHS, samples_per_epoch=len(train_lines), callbacks=None)

