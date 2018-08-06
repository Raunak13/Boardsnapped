import numpy as np
import cv2
import os

def ind(num):
	if num<10:
		out='0000'+str(num);
	elif(num<100):
		out='000'+str(num);
	elif(num<1000):
		out='00'+str(num);
	elif(num<10000):
		out='0'+str(num);
	else:
		out=str(num);
	return out;

x_folder='separated/'
dest='renamed/'
os.makedirs('renamed')
for i in range(1,51):
	writeto=dest+'lec'+str(i)
	os.makedirs(writeto)
	if(i==27):
		continue;
	src='lec'+str(i);
	image_folder=x_folder+src;
	for img in os.listdir(image_folder):
		image=cv2.imread(os.path.join(image_folder,img));
		index=ind(int(img[5:-4]));
		caption='l'+str(i)+'s'+index+'.jpg'
		cv2.imwrite(os.path.join(writeto,caption),image);		
	print(src+"loaded");