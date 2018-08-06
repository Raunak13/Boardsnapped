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

def format_data(in_x,in_y,index,timestep,batch_size):
	total_samples=in_x.shape[0];
	dim=(batch_size,timestep,in_x.shape[1],in_x.shape[2],in_x.shape[3])
	print(dim)
	x_train=np.zeros(batch_size*timestep*in_x.shape[1]*in_x.shape[2]*in_x.shape[3]).reshape(*dim);
	y_train=np.zeros(batch_size);
	
	offset=int(timestep/2);
	print('offset= '+str(offset));
	for i in range(dim[0]):
		x_train[i]=in_x[i+(index*batch_size):i+(index*batch_size)+timestep];
		y_train[i]=in_y[i+(index*batch_size)+offset];
	print(x_train.shape);
	print(y_train.shape);
	return x_train,y_train;



def getmodel(timesteps):
	model=Sequential();
	model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'),input_shape=(timesteps,224, 224, 3)));
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))));
	model.add(TimeDistributed(Conv2D(32, (3, 3),activation='relu',kernel_regularizer = regularizers.l2('0.5'))));
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Conv2D(64, (3, 3),activation='relu',kernel_regularizer = regularizers.l2('0.5'))));
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Conv2D(128, (3, 3),activation='relu',kernel_regularizer = regularizers.l2('0.5'))));
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Conv2D(64, (3, 3),activation='relu',kernel_regularizer = regularizers.l2('0.5'))));
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))));
	model.add(TimeDistributed(Flatten()));
	model.add(Bidirectional(LSTM(32,activation='tanh',return_sequences=False, kernel_regularizer=regularizers.l2('0.5')),merge_mode='concat'));
	model.add(Dense(1,activation='sigmoid'));
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']);
	model.summary();
	return model;




x_folder='../../data/separated/';
y_folder='../../data/annotated/';
print("loading data");
x_data,y_data,x_paths=load_data(x_folder,y_folder,21,25);
print("data loaded");
print(x_data.shape);
print(y_data.shape);
timesteps=20;
lrcn=getmodel(timesteps);
total_samples=x_data.shape[0];
iter=total_samples-timesteps;
batch_size=32;
epochs=2;
for e in range(epochs):
	for i in range(iter/batch_size):
		x_train, y_train= format_data(x_data,y_data,i,timesteps,batch_size);
		print('training on batch '+str(i)+' for epoch number '+str(e))
		lrcn.train_on_batch(x_train,y_train);

x_t,y_t,x_tpaths=load_data(x_folder,y_folder,26,26);
x_test,y_test=format_data(x_t,y_t,20);
y_pred=lrcn.predict(x_test,batch_size=32);
print(y_pred);
np.savetxt('y.csv',y_pred,delimiter=',',fmt='%.6f')


