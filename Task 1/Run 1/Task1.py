#Training Part

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras import optimizers
from keras import backend as K
import sklearn
from keras.models import load_model
import pandas as pd  
from keras.preprocessing import image
from PIL import Image
from libtiff import TIFF
import os
import keras
import keras.utils
from keras import utils as np_utils
from PIL import Image

#from sklearn.cross_validation import train_test_split10, 800
#import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import cv2
import csv
from cv2 import *


# dimensions of our images.
img_width, img_height = 256,256
target_size=(img_width, img_height)


#########################
### Tunables
#########################
train_data_dir = "/home/bhuvanaj/satellite/Task1/NITD/"
test_data_dir = "/home/bhuvanaj/satellite/Task1/test/"
train_path="/home/bhuvanaj/satellite/Task1/NITD/"

training_name1=os.listdir(train_data_dir)
for training_name in training_name1:
	print("Train Name")
	print(training_name)
	dir = os.path.join(train_path, training_name)
	print (dir)
	dir = dir + "/"
	print (dir)
	current_label = training_name
    #print("cl",current_label)
	train_dir = os.listdir(dir)
	for img_name in train_dir:	
		file=dir  + str(img_name)
		image = cv2.imread(file)
		if image.all():
			
			image = cv2.resize(image, (256, 256))
      #image = img_to_array(image)
      #x_train.append(image)
      #train_label.append(current_label)   # newly added
      #i=i+1






nb_train_samples = 2673
nb_test_samples = 60
epochs =10
batch_size = 60
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#########################
### Setup the model
#########################



model = Sequential()
model.add(Conv2D(64, (3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3,3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(4, 4)))

model.add(Conv2D(8, (4,4)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(4, 4)))

model.add(Conv2D(2, (2,2)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Activation('relu'))
model.add(Dropout(0.35))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(2, W_regularizer=keras.regularizers.l2(0.02)))
model.add(keras.layers.BatchNormalization())
model.add(Activation('softmax'))



sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(keras.optimizers.Adam(lr=1e-5), 'categorical_crossentropy', metrics=['accuracy'])

model.summary()


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


#########################
### Setup the generators
###fits the model on batches with real-time data augmentation
#########################


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#########################
### Build the model
#########################

'''model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs) '''


model.fit_generator(train_generator, 
	steps_per_epoch = nb_train_samples // batch_size, 
    epochs = epochs, validation_data = test_generator, 
    validation_steps = nb_test_samples // batch_size) 


#score = model.evaluate_generator(validation_generator, nb_val_samples/batch_size, workers=12)
#########################
### Performance evaluation
#########################
score = model.evaluate_generator(train_generator,nb_train_samples/batch_size)
print(" Total: ", len(train_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])


#########################
###Saving model for future use
#########################
model.save('task1.h5')



#########################
#EVALUATING MODEL
#########################


