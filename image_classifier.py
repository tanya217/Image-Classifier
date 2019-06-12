import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

l = {}
dir1 = '/home/malathy/Desktop/samplann'
#extract the names of the classes present under <object></object> and store it in a dictionary with file name as key and list of objects as values
for j in os.listdir(dir1):
	tree = ET.parse(dir1 + "/" + j)
	root = tree.getroot()
	temp=[]
	for c in root:
		if c.tag=="object":
			
			for i in c:
				if i.tag=="name":
					
					temp.append(i.text)
	l[j]=temp

print(l)
u=[]

#follow the order even in softmax
list1=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

#develop a one hot representation to tell which objects are present in each image
for a in l:
	v=[]
	for i in range(20):
		v.append(0)
	temp=l[a]
	
	for b in temp:
		c=list1.index(b)
		v[c]=1
	u.append(v)
print(u)
	
image_list=[]
n=0
dir2 = 	'/home/malathy/Desktop/Sample'
for d in os.listdir(dir2):
	image_arr=cv2.imread(dir2+ '/' +d)
	n=n+1
	image_list.append(image_arr)

print(image_list)
datanew=[]
max_h=500
max_w=500
s=(500,500,3)
arr_new=np.zeros(s,dtype=np.int)
print(arr_new)
final_arr=[]
for key in range(0,n):

	arr_new=np.zeros(s,dtype=np.int)
	#print(datanew[key].dtype)   
	arr_new[:image_list[key].shape[0],:image_list[key].shape[1],:]=image_list[key]
	
	final_arr.append(arr_new)

print(final_arr)


#y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_train=np.array(final_arr).astype(dtype='f')

y_train=np.array(u).astype(dtype='f')


#x_test = np.random.random((20, 100, 100, 3))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
image_test=[]
l_t = {}
dir4 = '/home/malathy/Desktop/testann'
#extract the names of the classes present under <object></object> and store it in a dictionary with file name as key and list of objects as values
for j in os.listdir(dir4):
	tree = ET.parse(dir4 + "/" + j)
	root = tree.getroot()
	temp=[]
	for c in root:
		if c.tag=="object":
			
			for i in c:
				if i.tag=="name":
					
					temp.append(i.text)
	l_t[j]=temp


dir3 = 	'/home/malathy/Desktop/testimage'
p=0
for e in os.listdir(dir3):
	image_arr1=cv2.imread(dir3+ '/' +e)
	p=p+1
	image_test.append(image_arr1)
	
print("---------------------------------------------------------------------------------------------")
print(image_test)
w=[]	
for a in l_t:
	v=[]
	for i in range(20):
		v.append(0)
	temp=l_t[a]
	
	for b in temp:
		c=list1.index(b)
		v[c]=1
	w.append(v)
	

final_arr_test=[]
for key1 in range(0,p):

	arr_new=np.zeros(s,dtype=np.int)
	
	arr_new[:image_test[key1].shape[0],:image_test[key1].shape[1],:]=image_test[key1]
	final_arr_test.append(arr_new)


x_test=np.array(final_arr_test).astype(dtype='f')
y_test=np.array(w).astype(dtype='f')


######################################################################################################################################
l_d = {}
dir5 = '/home/malathy/Desktop/samplann'

cor_list=[]
for j in os.listdir(dir5):
	tree = ET.parse(dir5 + "/" + j)
	root = tree.getroot()
	temp=[]
	for c in root:
		if c.tag=="object":
			
			for i in c:
				if i.tag=="bndbox":
					
					for ll in i:
						if(len(temp)==4):
							break;
						elif(ll.tag=="xmin"):
							temp.append(int(ll.text))
							continue
						elif(ll.tag=="ymin"):
							temp.append(int(ll.text))
							continue
						
						elif(ll.tag=="xmax"):
							temp.append(int(ll.text))
							continue
						elif(ll.tag=="ymax"):
							temp.append(int(ll.text))
							continue
						
						
						
					
	l_d[j]=temp
	cor_list.append(temp)
l_dtest={}
cor_test=[]
dir6='/home/malathy/Desktop/testann'
for j in os.listdir(dir6):
	tree = ET.parse(dir6 + "/" + j)
	root = tree.getroot()
	temp=[]
	for c in root:
		if c.tag=="object":
			
			for i in c:
				if i.tag=="bndbox":
					
					for ll in i:
						if(len(temp)==4):
							break;
						elif(ll.tag=="xmin"):
							temp.append(int(ll.text))
							continue
						elif(ll.tag=="ymin"):
							temp.append(int(ll.text))
							continue
						
						elif(ll.tag=="xmax"):
							temp.append(int(ll.text))
							continue
						elif(ll.tag=="ymax"):
							temp.append(int(ll.text))
							continue
						
						
						
					
	l_dtest[j]=temp
	cor_test.append(temp)
	
print("##########################################")
print(cor_list)
print("##########################################")
def classifier(x_train,y_train,x_test,y_test):
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))




	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(20, activation='softmax'))
	model.add(Dropout(0.5))
	#model.add(Dense(20, activation='sigmoid'))
	#model.add(Dense(20, activation='softmax'))




	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=1,validation_data=(x_test,y_test))
	return model



classifier1=classifier(x_train,y_train,x_test,y_test)
'''
score, acc,precision,recall = classifier1.evaluate(x_test, y_test)
                            
print('Score:', score)
print('Accuracy:', acc)
print('Precision',precision)
print('Recall',recall)
'''

def detector(x_train,y_train,x_test,y_test):
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))




	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	#model.add(Dense(20, activation='softmax'))
	#model.add(Dropout(0.5))
	model.add(Dense(32, activation='sigmoid'))
	#model.add(Dense(20, activation='softmax'))
	model.add(Dense(16,activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(16,activation='relu'))
	model.add(Dropout(0.25))
	#model.add(Flatten())
	model.add(Dense(8,activation='relu'))
	model.add(Dropout(0.25))
	#model.add(Flatten())
	model.add(Dense(4,activation='relu'))
	
	




	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=1,validation_data=(x_test,y_test))
	return model


detect1=detector(x_train,cor_list,x_test,cor_test)





