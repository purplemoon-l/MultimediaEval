# -*- coding: utf-8 -*-
"""Vgg19-TT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i18C_uz3OZL7Mzfc08uoICKeYEHom9Mk


!pip install PyDrive

import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

download = drive.CreateFile({'id': '1kKK2cwl7T7FIc2BQ8N5TPRUOF4kAikUw'})
download.GetContentFile('DOWNLOAD.zip')

!unzip DOWNLOAD.zip
"""
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image

#from keras.applications.imagenet_utils import preprocess_input

#from imagenet_utils import decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input
#from imagenet_utils import preprocess_input
#from keras.applications.vgg16 import preprocess_input

model = VGG19(weights='imagenet', include_top=False)

import os
import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras import optimizers 
from keras.callbacks import TensorBoard,ModelCheckpoint
import argparse
from time import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import cv2
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
#import preprocess_input


img_size=224
#ap = argparse.ArgumentParser()
#ap.add_argument("-train","--train_dir",type=str, required=True,help="(required) the train data directory")
#ap.add_argument("-num_class","--class",type=int, required=True,help="(required) number of classes to be trained")
#ap.add_argument("-val","--val_dir",type=str, required=True,help="(required) the validation data directory")



train_dir = '/home/bhuvanaj/satellite/Task1/NITD'
test_dir = '/home/bhuvanaj/satellite/Task1/test'

batch_size=32

IMAGE_SIZE    = (256, 256)
NUM_CLASSES   = 3
BATCH_SIZE    = 32  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 40  # freeze the first this many layers for training
epochs    = 30
#WEIGHTS_FINAL = 'model-inception_resnet_v2-final.h5'


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_dir,
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = valid_datagen.flow_from_directory(test_dir,
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

print('loading the model and the pre-trained weights...')
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
base_model = VGG19(include_top=False, weights='imagenet')
#base_model =InceptionResNetV2(include_top=False,
#                        weights='imagenet',
#                        input_tensor=None,
#                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
#x = net.output
## Here we will print the layers in the network
i=0
for layer in base_model.layers:
    layer.trainable = False
    i = i+1
    print(i,layer.name)
#sys.exit()
          ###########

x = base_model.output
#x = Dense(128)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)


##### Step-5:
############ Specify the complete model input and output, optimizer and loss

#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#filepath = 'vgg16_svm_model.h5'
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_best_only=True,save_weights_only=False, mode='min',period=1)
#callbacks_list = [checkpoint,tensorboard]


model = Model(inputs=base_model.input, outputs=predictions)

#model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9),metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(),metrics=["accuracy"])
#model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),metrics=["accuracy"])


num_training_img=2673
num_validation_img=60
stepsPerEpoch = num_training_img/batch_size
validationSteps= num_validation_img/batch_size

model.summary()

history=model.fit_generator(
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=30,
#        callbacks = callbacks_list,
        validation_data = validation_generator,
        validation_steps=validationSteps
)
#           train_batches,
#                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
#                        validation_data = valid_batches,
#                        validation_steps = valid_batches.samples // BATCH_SIZE,
#                        epochs = NUM_EPOCHS 

model.save('vgg19-TT-.h5')

score = model.evaluate_generator(validation_generator,num_validation_img/batch_size)
print(" Total: ", len(validation_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])

#lets plot the train and val curve
import matplotlib.pyplot as plt

#get the details form the history object
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']																																																																																																																																																																																																																																																	

epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Testing accurarcy')
plt.title('Training and Testing accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Test loss')
plt.title('Training and Testing loss')
plt.legend()

plt.show()

import os
target_size=( 256,256)
f1 = open("vgg19-TT-.csv",'w')
print("file opened", f1)
for root, dirs, files in os.walk("/home/bhuvanaj/satellite/Task1/NITDtest/", topdown=False):
    print("dirs", dirs)
    print("r", root)
    if root == "/home/bhuvanaj/satellite/Task1/NITDtest/":
        for name in dirs:
            print(name)
            TEST_DIR="/home/bhuvanaj/satellite/Task1/NITDtest/"+name+"/"  
            print(TEST_DIR)
            img_file=os.listdir(TEST_DIR)
            for f in (img_file):
                img = Image.open(TEST_DIR+f)
                img = cv2.imread(TEST_DIR+f)
                img= cv2.resize(img, target_size)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                #x = x/255.0
                #print("input shape",x.shape)
                #preds = custom_vgg_model.predict_classes(x) 
                preds = model.predict(x)
               # preds=preds.ravel()
                #print(preds)
                ptemp=np.argmax(preds,axis=1) 
                #print(ptemp)
                #print(preds[0])
                #print("prediction over")
                ptemp1=str(ptemp).replace("[", "")
                ptemp2=ptemp1.replace("]", "")
                print("actual:"+ name +" "+ ptemp2)
                f1.write(f+"\t"+ str(ptemp2) +"\n")
f1.close()
