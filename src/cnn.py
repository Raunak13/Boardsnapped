import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Conv3D, TimeDistributed, Dropout, Flatten, MaxPooling2D, MaxPool3D, Reshape, LSTM, Bidirectional
from keras import regularizers

def load_data(x_folder,y_folder,start,end):
	x_data=[];
	x_paths=[];
	y_data=None;
	for i in range(start,51):
		src='lec'+str(i);
		image_folder=x_folder+src;
		for img in sorted(os.listdir(image_folder)):
			image=cv2.resize(cv2.imread(os.path.join(image_folder,img)),(224,224));
			x_data.append(image);
			x_paths.append(str(os.path.join(image_folder,img)));
		label=np.loadtxt((y_folder+src+'.csv'),dtype=int);
		y_data = np.concatenate([y_data, label]) if y_data is not None else label;
		print(src+"loaded");
		if i==end:
			break;
	return np.array(x_data),np.array(y_data),x_paths;


def getmodel():
	model=Sequential();
	model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(224, 224, 3)));
	model.add(MaxPooling2D(pool_size=(2, 2)));
	model.add(Conv2D(32, (3, 3),activation='relu',kernel_regularizer = regularizers.l2('0.5')));
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3),activation='relu',kernel_regularizer = regularizers.l2('0.5')));
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3),activation='relu',kernel_regularizer = regularizers.l2('0.5')));
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3),activation='relu',kernel_regularizer = regularizers.l2('0.5')));
	model.add(MaxPooling2D(pool_size=(2, 2)));
	model.add(Flatten());
	model.add(Dense(1,activation='sigmoid'));
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']);
	model.summary();
	return model;



# def main():
x_folder='../data/separated/';
y_folder='../data/annotated/';
print("loading data");
x_data,y_data,x_paths=load_data(x_folder,y_folder,21,25);
print("data loaded");
print(x_data.shape);
print(x_paths);
print(y_data.shape);
cnn=getmodel();
cnn.fit(x_data,y_data,batch_size=128,epochs=200, validation_split=0.3);

# if __name__ == '__main__':
# 	main();


