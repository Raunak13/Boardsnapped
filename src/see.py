import os
import numpy as np
import cv2

threshold=0.85
src='renamed/'
key='misclassified/key/'
allkey='allkey/'
nokey='misclassified/nokey/'
for l in range(5,10):
	folder=src+'lec'+str(l)
	target=np.loadtxt('test/y_predicted'+str(l)+'.csv',delimiter=',')[:,0];
	pred=np.loadtxt('test/y_predicted'+str(l)+'.csv',delimiter=',')[:,1];
	key_save=key+'lec'+str(l)
	nokey_save=nokey+'lec'+str(l)
	os.makedirs(key+'lec'+str(l));
	os.makedirs(nokey+'lec'+str(l));
	os.makedirs(allkey+'lec'+str(l));
	i=0;
	for image in sorted(os.listdir(folder)):
		img=cv2.imread(os.path.join(folder,image));
		if(pred[i]>threshold):
			caption=str(i)+'prob'+str(pred[i])+'.jpg';
			cv2.imwrite(os.path.join((allkey+'lec'+str(l)),caption),img);
		if(target[i]-pred[i]>(1-threshold)):
			caption=str(i)+'prob'+str(pred[i])+'.jpg';
			cv2.imwrite(os.path.join(nokey_save,caption),img);
		elif (pred[i]-target[i]>threshold):
			caption=str(i)+'prob'+str(pred[i])+'.jpg';
			cv2.imwrite(os.path.join(key_save,caption),img);
		i=i+1;
	print(str(l)+'done')
