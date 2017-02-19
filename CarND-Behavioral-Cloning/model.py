import pickle, json, cv2
import numpy as np
import random
from scipy.misc import imread
from random import uniform
from keras.layers import Input, Activation, Dropout, Dense, Flatten, BatchNormalization 
from keras.models import Sequential, model_from_json, load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from preprocess import *


#data_file = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/example_data/driving_log.csv'
#image_path = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/data/'

data_file = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/track_data_new/driving_log.csv'

NUM_EPOCHS = 10
BATCH_SIZE = 8

def get_train_validation_path(data_file, validation_prob):
	train_lines = []
	validation_lines = []
	with open(data_file) as f:
		for i, line in enumerate(f):
			if i == 0:
				continue
			prob = uniform(0, 1)
			if prob >= validation_prob:
				train_lines.append(line.strip())
			else:
				validation_lines.append(line.strip())
	return train_lines, validation_lines
	
def get_generator(lines, image_path, batch_size):
	sample_size = len(lines)
	while 1:
		X_batch = []
		y_batch = []
		half_batch_size = int(batch_size)
		for start_i in range(0, sample_size, half_batch_size):
			#print("generate batch: ", i)
			end_i = start_i + half_batch_size 
			for line in lines[start_i:end_i]:
				image_tuples = process_line(line, image_path)
				for image_tuple in image_tuples:
					img = image_tuple[0]
					y = image_tuple[1]
					preprocessed = preprocess(img)				
					X_batch.append(preprocessed)
					y_batch.append(y)

					#flipped = flip_img(preprocessed)
					#X_batch.append(flipped)
					#y_batch.append(-y)
			yield np.array(X_batch), np.array(y_batch)

def get_nvidia_model():

	kernel_size = (3, 3)
	
	model = Sequential()
	
	model.add(BatchNormalization(mode=2, axis=1, input_shape=(ROWS, COLS, CHANNELS)))

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
	[train_lines, validation_lines] = get_train_validation_path(data_file, 0.2)
	
	random.shuffle(train_lines) 
	random.shuffle(validation_lines) 

	model = get_nvidia_model()
	#model = get_model()
	#model.summary()
	
	''' test generator
	generator = get_generator(train_lines, image_path, 1)
	x, y = next(generator)
	print("x: ", x.shape)
	print("y: ", y)

	for image in x:
		print(image.shape)
		cv2.namedWindow("test_generator", cv2.WINDOW_NORMAL)
		cv2.imshow("test_generator", image[:, :, 0])
		cv2.waitKey(0)

	'''	

	model.fit_generator(get_generator(train_lines, image_path, BATCH_SIZE),
		nb_epoch=NUM_EPOCHS, samples_per_epoch=3*len(train_lines), nb_val_samples=3*len(validation_lines),
		validation_data=get_generator(validation_lines, image_path, BATCH_SIZE), max_q_size=5)

	model.save('saved_model/model.h5')
	print("Saved model to disk")

	#model.fit_generator(get_generator(train_lines, image_path, BATCH_SIZE), 
	#	nb_epoch=NUM_EPOCHS, samples_per_epoch=len(train_lines), callbacks=None)
