import pickle, json, cv2
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from random import uniform
from keras.layers import Input, Activation, Dropout, Dense, Flatten, BatchNormalization 
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

def toRank3(gray_image):
	result = np.zeros((gray_image.shape[0], gray_image.shape[1], 1), dtype=gray_image.dtype)
	result[:, :, 0] = gray_image
	return result

def preprocess(image):
	resized_image = cv2.resize(image, (160, 80))
	img_yuv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
	img_y = img_yuv[:, :, 1]
	img_y = np.float32(img_y)
	img_y = img_y - np.mean(img_y)
	return toRank3(img_y)

def get_train_validation_path(data_file, image_path, validation_prob):
	train_lines = []
	validation_lines = []
	with open(data_file) as f:
		for i, line in enumerate(f):
			if i == 0:
				continue
			prob = uniform(0, 1)
			if prob >= 0:
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
			print("generate batch: ", i)
			i = i + 1
			end_i = start_i + batch_size
			for line in lines[start_i:end_i]:
				x, y = process_line(line, image_path)
				X_batch.append(x)
				y_batch.append(y)
			yield np.array(X_batch), np.array(y_batch)

def get_nvidia_model():
	model = Sequential()
	
	model.add(Convolution2D(24, 5, 5, activation='relu', input_shape=(80, 160, 1)))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(36, 5, 5, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(48, 5, 5, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, 5, 5, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	model.add(Dense(1164, activation='relu'))

	model.add(Dense(100, activation='relu'))

	model.add(Dense(50, activation='relu'))

	model.add(Dense(10, activation='relu'))

	#model.add(Dropout(0.25))
	model.add(Dense(1, activation='linear'))
	#model.add(Dense(1, init='glorot_normal', activation='linear'))

	model.compile('adam', 'mean_squared_error', ['accuracy'])
	return model

def get_model():
	lr = 0.0001
	weight_init='glorot_normal'
	opt = RMSprop(lr)
	loss = 'mean_squared_error'
	ROWS = 80
	COLS = 160
	CHANNELS = 3

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



data_file = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/data/TEST_IMG/driving_log.csv'
image_path = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/data/'
num_epoch = 10
batch_size = 32

[train_lines, validation_lines] = get_train_validation_path(data_file, image_path, 0.2)

model = get_nvidia_model()
model.summary()

#model.fit_generator(get_generator(train_lines, image_path, batch_size),
#	nb_epoch=num_epoch, samples_per_epoch=len(train_lines), nb_val_samples=len(validation_lines),
#	validation_data=get_generator(validation_lines, image_path, batch_size))

model.fit_generator(get_generator(train_lines, image_path, batch_size),
	nb_epoch=num_epoch, samples_per_epoch=len(train_lines))

img1 = imread(image_path + 'IMG/center_2016_12_01_13_32_43_457.jpg').astype(np.float32)
img2 = imread(image_path + 'IMG/center_2016_12_01_13_32_49_008.jpg').astype(np.float32)
preprocessed_img1 = preprocess(img1)
preprocessed_img2 = preprocess(img2)

print(preprocessed_img1)

cv2.imshow("img", preprocessed_img2[:, :, 0])
cv2.imshow("preprocessed_img", preprocessed_img1[ :, :, 0])
cv2.waitKey(0)

#model.fit(preprocessed_img1[None, :, :, :], np.array([0.0617599]), nb_epoch=num_epoch, batch_size=batch_size)

steering_angle1 = float(model.predict(preprocessed_img1[None, :, :, :], batch_size=1))
steering_angle2 = float(model.predict(preprocessed_img2[None, :, :, :], batch_size=1))

print ('steering angle: ', steering_angle1, steering_angle2)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
