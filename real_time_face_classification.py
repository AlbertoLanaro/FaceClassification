#!/Users/albertolanaro/venv3/bin/python3
import numpy as np
import cv2
import keras
from keras.models import model_from_json
import os

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_model():
	# load json and create model
	json_file = open('trained_models/trained_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("trained_models/model_weights.h5")
	print("Model loaded from disk")
	loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam', metrics=['accuracy'])

	return loaded_model

def detect_face(frame, face_classifier):
	# detect face and classification
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.6, 8)
	if len(faces) > 0:
		# extract face from frame
		face_imgs = []
		for (x,y,w,h) in faces:
			face_temp = gray[y:y+h,x:x+w]
			face_temp = cv2.resize(face_temp, (30,30), interpolation=cv2.INTER_CUBIC) / 255
			face_imgs.append(face_temp)
		face_imgs = np.array(face_imgs).reshape((-1, 30, 30, 1))
		return face_imgs, faces
	else:
		return [], faces

'''
def isFace(faces, eyes):
	correct_face = []
	n_eyes = 0
	for (x,y,w,h) in faces:
		for (ex,ey,_,_) in eyes:
			if ((ex>(x-w//2)) and (ex<(x+w//2)) and (ey<(y+h//2)) and (ey>(y-h//2))):
				print("x:%d y:%d ex:%d ey:%d" % (x,y,ex,ey))
				n_eyes += 1
		if n_eyes == 2:
			correct_face.append((x,y,w,h))
			n_eyes = 0

	return correct_face
'''

def display_result(img, faces, name):
	for ((x,y,w,h), n)  in zip(faces, name):
			cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
			cv2.putText(img, str(n), (x+w//10,y-h//10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
	# display results
	cv2.imshow('Real-time Face classification', img)
	cv2.waitKey(1)

def face_classification(face_imgs, cl):
	temp_pred = cl.predict(face_imgs)
	
	return np.argmax(temp_pred, axis=1), np.max(temp_pred, axis=1)

# maps labels in user's name
def map_label(labels):
	PATH = 'Faces/'
	usr = [item for item in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, item))]
	usr_name = [i.split('_')[0] for i in usr]
	mapped = [usr_name[i] for i in labels]

	return mapped 
	
if __name__ == '__main__':
	# face detector
	face_cascade = cv2.CascadeClassifier('/Users/albertolanaro/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
	#load trained CNN model
	cl = load_model()
	# webcam
	cap = cv2.VideoCapture(0)
	predicted_proba = 0.0
	predicted_name = ''
	while True:
		grabbed, frame = cap.read()	
		if grabbed:
			face_imgs, faces = detect_face(frame, face_cascade)
			if len(faces) > 0:
				predicted_labels, predicted_probas = face_classification(face_imgs, cl)
				predicted_name = map_label(predicted_labels)
				print('------------')
				for i,j in zip(map_label(predicted_labels), predicted_probas):	
					print('Detected %s w.p. %f' % (i,j))
			display_result(frame, faces, predicted_name)
