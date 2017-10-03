#!/Users/albertolanaro/venv3/bin/python3
import sys
if len(sys.argv) != 2:
	print('usage: ./face_classification.py <# training epochs>')
	sys.exit(1)
import os
import numpy as np
import cv2
import glob
from random import shuffle
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten 
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json

def load_imgs(path):

	usr = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
	DEFAULT_SIZE = (30,30)
	imgs = []
	for i in usr:
		imgs.append([cv2.resize(cv2.imread(j, cv2.IMREAD_GRAYSCALE), DEFAULT_SIZE, interpolation=cv2.INTER_CUBIC) for j in glob.glob(PATH + i + '/*.png')])

	print('Number of users:', len(imgs))
	for i,j in enumerate(imgs):
		print('\tSamples for user %d: %d' % (i+1,len(j)))

	return imgs, usr

def create_labels(imgs):
	y = []
	for i in range(len(imgs)):
		y.append(i * np.ones(len(imgs[i])))
	y = np.hstack(y)

	return y

def reshape_for_keras(data):
	temp = np.vstack(data)
	
	return temp.reshape(temp.shape + (1,))

def train_test_split(data, test_size):
	rnd_index = np.arange(data.shape[0])
	shuffle(rnd_index)
	last_test_index = round(test_size * len(rnd_index))
	test_index =  rnd_index[:last_test_index]
	train_index = rnd_index[last_test_index:]

	return train_index, test_index

def create_model(data, n_classes):
	input_data = Input(shape=data.shape[1:])
	feat_layer0 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(input_data)
	x = MaxPooling2D(pool_size=(2,2))(feat_layer0)
	x = Dropout(0.25)(x)
	feat_layer1 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2,2))(feat_layer1)
	x = Dropout(0.5)(x)
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	output_layer = Dense(n_classes, activation='softmax')(x) 
	# Model for prediction
	cl = Model(input_data, output_layer)
	cl.summary()

	cl.compile(loss=keras.losses.categorical_crossentropy, 
			   optimizer='Adam', 
			   metrics=['accuracy'])
	# Model for feature extraction
	feat_extractor = Model(input_data, feat_layer0)

	return cl, feat_extractor

def train_and_evaluate(cl, x_train, y_train, x_test, y_test, n_classes):
	batch_size = 64
	epochs = int(sys.argv[1])

	y_trainCNN = keras.utils.to_categorical(y_train, n_classes)
	y_testCNN = keras.utils.to_categorical(y_test, n_classes)

	cl.fit(x_train, y_trainCNN, batch_size=batch_size, epochs=epochs, verbose=1)
	pred = cl.predict(x_test)
	y_pred = np.argmax(pred, axis=1)
	conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_train))
	accuracy = 1-np.sum(np.abs(y_pred - y_test)) / len(y_test)

	print('Confusion matrix:\n', conf_matrix)
	print('Test accuracy:', accuracy)

def save_model(model):
	model_json = model.to_json()
	with open("trained_models/trained_model.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("trained_models/model_weights.h5")
	print("Saved model to disk")

if __name__ == '__main__':

	PATH = 'Faces/'
	# load images in grayscale
	imgs, usr = load_imgs(PATH)
	# create labels
	yy = create_labels(imgs)
	# reshape data in keras format
	xx = reshape_for_keras(imgs)
	# normalize data
	xx = xx/255 
	# divide train/test
	train_index, test_index = train_test_split(xx, test_size=0.2)
	x_train = xx[train_index]
	y_train = yy[train_index]
	x_test = xx[test_index]
	y_test = yy[test_index]
	# print useful info
	print('Total number of samples:', xx.shape[0])
	print('\tNumber of training samples:', x_train.shape[0])
	print('\tNumber of test samples:', x_test.shape[0])
	# create keras model
	cl, feat_extractor = create_model(xx, len(usr))
	train_and_evaluate(cl, x_train, y_train, x_test, y_test, len(usr))
	# save trained model
	save_model(cl)
	# display some features extracted from the CNN
	random_example = np.random.randint(low=0, high=len(yy)-1, size=20)
	feat_example = feat_extractor.predict(xx[random_example])
	original_pic = xx[random_example]
	for i in np.arange(feat_example.shape[0]):
		temp0 = cv2.resize(feat_example[i,:,:,10], (300,300), interpolation=cv2.INTER_CUBIC)
		temp1 = cv2.resize(original_pic[i], (300,300), interpolation=cv2.INTER_CUBIC)
		cv2.imshow(str(i), temp0)
		cv2.imshow(str(i) + '_orig', temp1)
		cv2.waitKey(0)

