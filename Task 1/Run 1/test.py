from keras import optimizers
import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2

# dimensions of our images
train_path="/home/bhuvanaj/satellite/Task1/NITDtest/images/"


    #print("cl",current_label)
train_dir = os.listdir(train_path)
for img_name in train_dir:	
	file=train_path + str(img_name)
	imag = cv2.imread(file)
	if imag.all():
		imag = cv2.resize(imag, (256, 256))
img_width, img_height = 256 , 256

# load the model we saved
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model = load_model('task1.h5')
model.compile(keras.optimizers.Adam(lr=1e-6), 'categorical_crossentropy', metrics=['accuracy'])

test_dir = "/home/bhuvanaj/satellite/Task1/NITDtest/images/"

f1 = open("/home/bhuvanaj/satellite/Task1/NITDtest/results_2.csv",'w')
files = [f for f in sorted(os.listdir("/home/bhuvanaj/satellite/Task1/NITDtest/images/"))]
os.chdir(test_dir)
for i in files:
	img = image.load_img(i, target_size=(img_width, img_height))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	images = np.vstack([x])
	classes = model.predict(images, batch_size=10)
	class_pred = model.predict_classes(images,batch_size=10)
	print (classes)
	print(classes)
	ptemp=str(classes).replace("[", "")
	ptemp1=ptemp.replace("]", "")
	

	temp=str(class_pred).replace("[", "")
	temp1=temp.replace("]", "")


	f1.write(i+"\t"+ ptemp1 +"\t" + temp1 +"\n")
f1.close()

