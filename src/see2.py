import numpy as np
accuracy=0;
sum_temp=0;
sum_l=0
lec_acc=np.zeros(9);
total=np.zeros(9)
precision =0
tot_pred = 0
for l in range(1,10):
	prec_l = 0
	if (l==2):
		continue
	target=np.loadtxt('test/y_predicted'+str(l)+'.csv',delimiter=',')[:,0];
	prob=np.loadtxt('test/y_predicted'+str(l)+'.csv',delimiter=',')[:,1];
	pred= (prob>.75)*1
	for i in range(pred.shape[0]):
		if(pred[i]==1):
			if(target[i]==1):
				prec_l+=1
			elif(target[i-1]==1):
				prec_l +=1
			elif(target[i-2]==1):
				prec_l +=1
			elif(target[i+1]==1):
				prec_l +=1
			elif(target[i+2]==1):
				prec_l +=1
	precision += prec_l
	tot_pred +=sum(pred)
	print ((prec_l+0.0)/sum(pred))

	# total[l-1]=(sum(pred.astype('int')==target.astype('int'))*1.0)/len(target)

	# correct=((pred.astype('int'))==(target.astype('int')))*1.0
	# c1=correct*target.astype('int')
	# # print(sum(c1)/sum(target));
	# sum_temp+=sum(c1)
	# sum_l+=sum(target)
	# t=0
	# for i in range(len(target)):
	# 	if target[i]==1:
	# 		d=sum(pred[i-2:i+3]);
	# 		if d>0:
	# 			t+=1
	# print(t/sum(target));
	# lec_acc[l-1]=t/sum(target);

print (precision+0.)/tot_pred
# accuracy=(sum_temp+0.0)/sum_l;
# print(accuracy)
# print( sum(lec_acc)/8.0)
# print(sum(total)/8.0)