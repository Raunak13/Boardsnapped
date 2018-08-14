import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import math
import os
import keras
from keras.utils import generic_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Conv3D, TimeDistributed, Dropout, Flatten, MaxPooling2D, MaxPool3D, Reshape, LSTM, Bidirectional
from keras import regularizers
import h5py
from keras.optimizers import Adam
from sklearn.metrics import classification_report

def load_data(x_folder,y_folder,start,end):
	x_data=[];
	x_paths=[];
	y_data=None;
	for i in range(start,51):
		if(i==27):
			continue;
		src='lec'+str(i);
		image_folder=x_folder+src;
		for img in sorted(os.listdir(image_folder)):
			image=cv2.resize(cv2.imread(os.path.join(image_folder,img)),(224,224));
			image = cv2.normalize(image, image,alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			x_data.append(image);
			x_paths.append(str(os.path.join(image_folder,img)));
		label=np.loadtxt((y_folder+src+'.csv'),dtype=int);
		y_data = np.concatenate([y_data, label]) if y_data is not None else label;
		
		print(src+"loaded");
		if i==end:
			break;
	y_data=np.reshape(y_data,(-1,1)).astype(int);
	return np.array(x_data),np.array(y_data),x_paths;

def format_train_data(in_x,in_y,timestep):
	n_samples=in_x.shape[0];
	repetition=0.5;
	n_batches=int((n_samples-timestep)/((1-repetition)*timestep))+1
	dim=(n_batches,timestep,in_x.shape[1],in_x.shape[2],in_x.shape[3])
	print("dim ="+str(dim));
	x_train=np.zeros(dim);
	y_train=np.zeros((n_batches,timestep,1)).astype(int);
	sample_wts=np.zeros((n_batches,timestep));
	index=0;
	for i in range (n_batches):
		# print(index)
		x_train[i]=in_x[index:index+timestep];
		y_train[i]=in_y[index:index+timestep];
		sample_wts[i,:]=y_train[i,:][:,0]*20+1;
		index=int(index+(1-repetition)*timestep);	
	print(x_train.shape);
	print(y_train.shape);
	return x_train,y_train,sample_wts;

def revert(y_p,timestep):
	length=y_p.shape[0]
	rep=0.5
	output=np.zeros(int((length-1)*(1-rep)*timestep+timestep));
	index=0;
	for i in range(length):
		output[index:index+timestep]=np.maximum(output[index:index+timestep],y_p[i][:,0]);
		index=int(index+(1-rep)*timestep);	
	return output;




def getmodel(timestep):
	model=Sequential();
	model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu',padding='same'),input_shape=(timestep,224, 224, 3)));
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))));
	model.add(TimeDistributed(Conv2D(32, (3, 3),activation='relu',padding='same',kernel_regularizer = regularizers.l2('0.2'))));
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	# model.add(TimeDistributed(Conv2D(64, (3, 3),activation='relu',kernel_regularizer = regularizers.l2('0.5'))));
	# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Conv2D(64, (3, 3),activation='relu',padding='same',kernel_regularizer = regularizers.l2('0.2'))));
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Conv2D(32, (3, 3),activation='relu',padding='same',kernel_regularizer = regularizers.l2('0.2'))));
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))));
	model.add(TimeDistributed(Flatten()));
	model.add(Bidirectional(LSTM(128,activation='tanh',return_sequences=True, kernel_regularizer=regularizers.l2('0.2')),merge_mode='concat'));
	model.add(TimeDistributed(Dense(100,activation='relu')));
	model.add(TimeDistributed(Dense(1,activation='sigmoid')))
	# model.compile(loss='mean_squared_error',optimizer='adam',sample_weight_mode='temporal');
	model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4),sample_weight_mode='temporal');
	model.summary();
	return model;

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


x_folder='../../data/renamed/';
y_folder='../../data/labels/';
print("loading data");
x_data,y_data,x_paths=load_data(x_folder,y_folder,31,32);
print("data loaded");
print(x_data.shape);
print(y_data.shape);
# y_faux=np.zeros(y_data.shape).astype('int');
# ind=np.random.randint(0,y_data.shape[0],int(y_data.shape[0]/3));
# y_faux[ind]=1;

timestep=16;
x_train, y_train, sample_wts= format_train_data(x_data,y_data,timestep);
lrcn=getmodel(timestep);
# lrcn.load_weights('')
# class_weights={0:1.,1:100};
batch_size=16;
n_epochs=200;
n_batches=int(np.ceil(float(x_train.shape[0]/batch_size)));
lrcn.fit(x_train,y_train,batch_size=batch_size,epochs=n_epochs,sample_weight=sample_wts);
train_loss=np.zeros((n_epochs,n_batches));
for e in range(n_epochs):
	progbar = generic_utils.Progbar(n_batches*batch_size)
	batch_counter = 1
	for b in range (n_batches):
		if ((b+1)*batch_size<x_train.shape[0]):
			x_batch=x_train[b*batch_size:(b+1)*batch_size];
			y_batch=y_train[b*batch_size:(b+1)*batch_size];
			# train_loss[e,b]=lrcn.train_on_batch(x_batch,y_batch);
			t = lrcn.train_on_batch(x_batch,y_batch);
		else:
			x_batch=x_train[b*batch_size:];
			y_batch=y_train[b*batch_size:];
			# train_loss[e,b]=lrcn.train_on_batch(x_batch,y_batch);
			t = lrcn.train_on_batch(x_batch,y_batch);
		batch_counter+=1
		progbar.add(batch_size, values=[("Loss:",t )])
	print ""
		# print("epoch = "+str(e)+" batch = "+str(b)+" loss: "+str(train_loss[e,b]))


lrcn.save('../outputs/lstm.h5');
lrcn.save_weights('../outputs/lstm_weights.h5')

# lrcn.load_weights('..outputs/lstm1_weights.h5')

for j in range(1,2):
	x_t,y_t,x_tpaths=load_data(x_folder,y_folder,j,j);
	x_test,y_test,y_sample_wt=format_train_data(x_t,y_t,timestep);
	y_pred=lrcn.predict(x_test,batch_size=4);
	y_form=np.reshape(revert(y_pred,timestep)(-1,1));
	y_target=y_t[0:len(y_form)]
	tosave=np.append(y_target,y_form,axis=1);
	y_p=((y_form>0.5)*1);
	target_names = ['non_key', 'key']
	print(classification_report(y_target, y_p, target_names=target_names));
	# print(y_pred);
	np.savetxt(('../outputs/'+str(j)+'ysave.csv'),tosave,delimiter=',',fmt='%.6f')




