import json, cv2, csv
import numpy as np
from keras.layers import Input, Convolution2D, Activation, Dropout, Dense, Flatten, Lambda
from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import Adam
from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

NUM_EPOCHS = 10
BATCH_SIZE = 8

# Load image file names
def load_samples(file_name):
	samples = []
	with open(file_name) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
	return samples

# Data generator
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
					# Returns three image and label pairs
					# for center, left and right cameras
					preprocessed_image = preprocess(image_tuple[0])				
					X_batch.append(preprocessed_image)
					y_batch.append(image_tuple[1])
					
					X_train = np.array(X_batch)
					y_train = np.array(y_batch)

			yield shuffle(X_train, y_train)

# Main model
def get_model():
	
	# Converlution kernal size
	kernel_size = (3, 3)
	
	model = Sequential()

	# Normalize images
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(ROWS, COLS, CHANNELS)))
	
	# Convolutional layers with depth: 8, 16, 24, 32, 32
	model.add(Convolution2D(8, kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))

	model.add(Convolution2D(16, kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))

	model.add(Convolution2D(24, kernel_size[0], kernel_size[1], border_mode='valid', activation='relu'))

	model.add(Convolution2D(32, kernel_size[0], kernel_size[1],  border_mode='valid', activation='relu'))
	
	model.add(Convolution2D(32, kernel_size[0], kernel_size[1],  border_mode='valid', activation='relu'))

	# Flatten Data
	model.add(Flatten())

	# Fully connected layers with neuron number: 100, 50, 20, 10, 1
	model.add(Dense(100, activation='relu'))

	# Add dropout to avoid overfitting
	model.add(Dropout(0.25))

	model.add(Dense(50, activation='relu'))

	model.add(Dense(20, activation='relu'))
	
	model.add(Dense(10, activation='relu'))

	model.add(Dense(1, activation='linear'))
	
	# Compile model with optimizer: Adam, Learning rate: 1e-4
	model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

	return model

# Visualize training loss and validation loss
def visualize_loss(history_object):
	# print the keys contained in the history object
	print(history_object.history.keys())

	# plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()

if __name__ == '__main__':

	# Load image file names to samples
	file_name = '/home/linfeng-zc/Documents/Udacity/CarND-Behavioral-Cloning/data/track_data_new/driving_log.csv'
	samples = load_samples(file_name)

	# Split train, validation and test data file path	
	train_samples, test_samples = train_test_split(samples, test_size=0.1)
	train_samples, validation_samples = train_test_split(train_samples, test_size=0.25)

	# Generators
	train_generator = get_generator(train_samples, BATCH_SIZE)
	validation_generator = get_generator(validation_samples, BATCH_SIZE)
	test_generator = get_generator(test_samples, BATCH_SIZE)
	
	# Get model
	model = get_model()
	model.summary()

	# Train model
	augement_factor = 3
	history_object = model.fit_generator(train_generator, nb_epoch=NUM_EPOCHS, samples_per_epoch=augement_factor * len(train_samples), 
		validation_data=validation_generator, nb_val_samples=augement_factor*len(validation_samples), max_q_size=5)

	# Test model
	print (model.evaluate_generator(test_generator, val_samples=augement_factor*len(test_samples), max_q_size=5))

	# Save model
	model.save('saved_model/model.h5')
	print("Saved model to disk")

	visualize_loss(history_object)
