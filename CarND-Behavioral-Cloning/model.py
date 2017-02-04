import pickle, json, cv2
import numpy as np
from scipy.misc import imread
from random import uniform
from keras.layers import Input, Activation, Dropout, Dense, Flatten, BatchNormalization 
from keras.models import Sequential, model_from_json, load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


#data_file = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/data/TEST_IMG/driving_log.csv'
#image_path = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/data/'

#data_file = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/example_data/driving_log.csv'
#image_path = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/data/'

data_file = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/track_data/driving_log.csv'
image_path = ''

NUM_EPOCHS = 10
BATCH_SIZE = 32

ROI_ROW = 60

ROWS = 20
COLS = 64
CHANNELS = 1


# Convert one channel 2D image to 3D image with dimention (rows, cols, 1)
def toRank3(gray_image):
	result = np.zeros((gray_image.shape[0], gray_image.shape[1], 1), dtype=gray_image.dtype)
	result[:, :, 0] = gray_image
	return result

# Preprocessing images
def preprocess(image):
	# To YUV
	img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

	# Crop ROI and take Y channel
	img_roi = image[ROI_ROW:, :, 1]

	# Resize image
	resized_roi = cv2.resize(img_roi, (COLS, ROWS))

	# Normalize image
	resized_roi = np.float32(resized_roi)
	resized_roi = (resized_roi - 128.0) / 128.0

	return toRank3(resized_roi)

def get_train_validation_path(data_file, image_path, validation_prob):
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
	
def process_line(line, image_path):
	data = line.split(',')
	img = imread(image_path + data[0]).astype(np.float32)
	preprocessed_img = preprocess(img)
	return preprocessed_img, data[3]

def get_generator(lines, image_path, batch_size):
	sample_size = len(lines) 
	while 1:
		X_batch = []
		y_batch = []
		i = 0
		for start_i in range(0, sample_size, batch_size):
			#print("generate batch: ", i)
			i = i + 1
			end_i = start_i + batch_size
			for line in lines[start_i:end_i]:
				x, y = process_line(line, image_path)
				X_batch.append(x)
				y_batch.append(y)
			yield np.array(X_batch), np.array(y_batch)

def get_nvidia_model():
	model = Sequential()
	
	model.add(BatchNormalization(mode=2, axis=1, input_shape=(ROWS, COLS, CHANNELS)))
	model.add(Convolution2D(24, 3, 3, border_mode='valid', activation='relu'))

	model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='relu'))

	model.add(Convolution2D(48, 3, 3, border_mode='valid', activation='relu'))

	model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))

	model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))

	model.add(Flatten())

	model.add(Dense(1164, activation='relu'))

	model.add(Dense(100, activation='relu'))

	model.add(Dense(50, activation='relu'))

	model.add(Dense(10, activation='relu'))

	#model.add(Dropout(0.25))
	model.add(Dense(1, activation='linear'))

	model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
	return model

def get_model():
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
	
	return model

#TODO
def three_imgs_test():
	img1 = imread(image_path + 'IMG/center_2016_12_01_13_32_43_457.jpg').astype(np.float32)
	img2 = imread(image_path + 'IMG/center_2016_12_01_13_32_49_008.jpg').astype(np.float32)
	preprocessed_img1 = preprocess(img1)
	preprocessed_img2 = preprocess(img2)


	cv2.imshow("img", preprocessed_img2[:, :, 0])
	cv2.imshow("preprocessed_img", preprocessed_img1[ :, :, 0])
	cv2.waitKey(0)

	steering_angle1 = float(model.predict(preprocessed_img1[None, :, :, :], batch_size=1))
	steering_angle2 = float(model.predict(preprocessed_img2[None, :, :, :], batch_size=1))

	print ('steering angle: ', steering_angle1, steering_angle2)

if __name__ == '__main__':
	[train_lines, validation_lines] = get_train_validation_path(data_file, image_path, 0.2)

	model = get_nvidia_model()
	#model.summary()

	model.fit_generator(get_generator(train_lines, image_path, BATCH_SIZE),
		nb_epoch=NUM_EPOCHS, samples_per_epoch=len(train_lines), nb_val_samples=len(validation_lines),
		validation_data=get_generator(validation_lines, image_path, BATCH_SIZE))

	#model.fit_generator(get_generator(train_lines, image_path, BATCH_SIZE), 
	#	nb_epoch=NUM_EPOCHS, samples_per_epoch=len(train_lines), callbacks=None)

	#model.fit(preprocessed_img1[None, :, :, :], np.array([0.0617599]), nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2)

	'''
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")

	print("Saved model to disk")
	'''
	
	model.save('model.h5')
