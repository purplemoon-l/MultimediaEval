
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import keras 
# dimensions of our images
img_width, img_height = 256 , 256

# load the model we saved
model = load_model('/home/bhuvanaj/satellite/Task2/task2_10epoch_48batch_6lr.h5')
model.compile(keras.optimizers.Adam(lr=1e-6), 'categorical_crossentropy', metrics=['accuracy'])



test_dir = '/home/bhuvanaj/satellite/Task2/Data/Testii/Test/'

f1 = open("/home/bhuvanaj/satellite/Task2/results.csv",'w')
files = [f for f in sorted(os.listdir("/home/bhuvanaj/satellite/Task2/Data/Testii/Test/"))]
for i in files:
	img = image.load_img(test_dir+i, target_size=(img_width, img_height))
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

'''
# predicting images
img = image.load_img('/home/bhuvanaj/satellite/Task3/Data/Overall/im_rescaled-0026.tif', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print("One image")
print (classes)

# predicting multiple images at once
img = image.load_img('/home/bhuvanaj/satellite/Task3/Data/Overall/im_rescaled-0024.tif', target_size=(img_width, img_height))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
images = np.vstack([x, y])
classes = model.predict_classes(images, batch_size=10)
print (classes)
'''
