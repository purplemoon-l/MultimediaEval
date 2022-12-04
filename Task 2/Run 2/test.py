from keras.applications.vgg16 import VGG16, preprocess_input
#from imagenet_utils import preprocess_input
#from keras.applications.vgg16 import preprocess_input

#model = VGG19(weights='imagenet', include_top=False)

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import keras
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
import os
model = load_model('/home/bhuvanaj/satellite/Task2/vgg19-TT-.h5')
target_size=( 256,256)
f1 = open("vgg19-TT-.csv",'w')
print("file opened", f1)
for root, dirs, files in os.walk("/home/bhuvanaj/satellite/Task2/Data/Testii/", topdown=False):
    print("dirs", dirs)
    print("r", root)
    if root == "/home/bhuvanaj/satellite/Task2/Data/Testii/":
        for name in dirs:
            print(name)
            TEST_DIR="/home/bhuvanaj/satellite/Task2/Data/Testii/"+name+"/"  
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
