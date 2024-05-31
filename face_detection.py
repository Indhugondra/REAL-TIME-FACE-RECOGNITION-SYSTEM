#!/usr/bin/env python
# coding: utf-8

# In[30]:


import cv2
import os

os.getcwd()
# for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# resolution of the webcam
screen_width = 640       # try 640 if code fails
screen_height = 720

# default webcam
stream = cv2.VideoCapture(0)

while(True):
    # capture frame-by-frame
    (grabbed, frame) = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)

    # for each faces found
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        color = (0, 255, 255) # in BGR
        stroke = 5
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    # show the frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):    # Press q to break out
        break                  # of the loop

# cleanupm
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)


# In[4]:


os.getcwd()


# In[1]:


get_ipython().system('pip install keras_vggface')


# In[6]:


get_ipython().system('pip install keras_applications')


# In[36]:


import cv2
import os
import pickle
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

headshots_folder_name = 'Headshots'

# dimension of images
image_width = 224
image_height = 224

# for detecting faces
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# set the directory containing the images
images_dir = os.path.join(".", headshots_folder_name)

current_id = 0
label_ids = {}

# iterates through all the files in each subdirectories
for root, _, files in os.walk(images_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
        # path of the image
            path = os.path.join(root, file)

            # get the label name (name of the person)
            label = os.path.basename(root).replace(" ", ".").lower()

            # add the label (key) and its number (value)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            # load the image
            imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
            image_array = np.array(imgtest, "uint8")

                # get the faces detected in the image
            faces = facecascade.detectMultiScale(imgtest,
                    scaleFactor=1.1, minNeighbors=5)

                # if not exactly 1 face is detected, skip this photo
            
            if len(faces) != 1:
                print(f'---Photo skipped---\n')
                # remove the original image
                os.remove(path)
                continue

                # save the detected face(s) and associate
                # them with the label
            for (x_, y_, w, h) in faces:

                    # draw the face detected
                face_detect = cv2.rectangle(imgtest,
                            (x_, y_),
                            (x_+w, y_+h),
                            (255, 0, 255), 2)
                plt.imshow(face_detect)
                plt.show()

                    # resize the detected face to 224x224
                size = (image_width, image_height)

                    # detected face region
                roi = image_array[y_: y_ + h, x_: x_ + w]

                    # resize the detected head to target size
                resized_image = cv2.resize(roi, size)
                image_array = np.array(resized_image, "uint8")

                    # remove the original image
                os.remove(path)

                    # replace the image with only the face
                im = Image.fromarray(image_array)
                im.save(path)


# In[8]:


len(faces)


# In[37]:


import os
import pandas as pd
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input


from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam


from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, AveragePooling2D, BatchNormalization, ZeroPadding2D, Convolution2D, MaxPooling2D, Activation 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# import transfer learning model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Input


# In[38]:


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
'./Headshots',
target_size=(224,224),
color_mode='rgb',
batch_size=32,
class_mode='categorical',
shuffle=True)


# In[39]:


train_generator.class_indices.values()
# dict_values([0, 1, 2])
NO_CLASSES = len(train_generator.class_indices.values())
NO_CLASSES


# In[40]:


from keras_vggface.vggface import VGGFace

base_model = VGGFace(include_top=True,
    model='vgg16',
    input_shape=(224, 224, 3))
base_model.summary()

print(len(base_model.layers))
# 26 layers in the original VGG-Face


# In[41]:


base_model = VGGFace(include_top=False,
model='vgg16',
input_shape=(224, 224, 3))
base_model.summary()
print(len(base_model.layers))
# 19 layers after excluding the last few layers


# In[42]:


x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# final layer with softmax activation
preds = Dense(NO_CLASSES, activation='softmax')(x)


# In[43]:


# create a new model with the base model's original input and the 
# new model's output
model = Model(inputs = base_model.input, outputs = preds)
model.summary()

# don't train the first 19 layers - 0..18
for layer in model.layers[:19]:
    layer.trainable = False

# train the rest of the layers - 19 onwards
for layer in model.layers[19:]:
    layer.trainable = True


# In[ ]:





# In[44]:


model.compile(optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# In[45]:


model.fit(train_generator,
  batch_size = 1,
  verbose = 1,
  epochs = 20)


# In[46]:


# creates a HDF5 file
model.save(
    'transfer_learning_trained' +
    '_face_cnn_model.h5')


# In[47]:


import pickle

class_dictionary = train_generator.class_indices
class_dictionary = {
    value:key for key, value in class_dictionary.items()
}
print(class_dictionary)


# In[49]:


# save the class dictionary to pickle
face_label_filename = 'face-labels.pickle'
with open(face_label_filename, 'wb') as f: pickle.dump(class_dictionary, f)


# In[50]:


import cv2
import os
import pickle
import numpy as np
import pickle

from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras_vggface import utils

# dimension of images
image_width = 224
image_height = 224

# load the training labels
face_label_filename = 'face-labels.pickle'
with open(face_label_filename, "rb") as \
    f: class_dictionary = pickle.load(f)

class_list = [value for _, value in class_dictionary.items()]
print(class_list)


# In[57]:


import keras.utils as image


# In[59]:


# for detecting faces
facecascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

for i in range(1,10): 
    test_image_filename = f'./facetest/face{i}.jpg'

# load the image
imgtest = cv2.imread(test_image_filename, cv2.IMREAD_COLOR)
image_array = np.array(imgtest, "uint8")

# get the faces detected in the image
faces = facecascade.detectMultiScale(imgtest, 
    scaleFactor=1.1, minNeighbors=5)

# if not exactly 1 face is detected, skip this photo
if len(faces) != 1: 
    print(f'---We need exactly 1 face;photo skipped---')
    print()

for (x_, y_, w, h) in faces:
    # draw the face detected
    face_detect = cv2.rectangle(
        imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)
    plt.imshow(face_detect)
    plt.show()

    # resize the detected face to 224x224
    size = (image_width, image_height)
    roi = image_array[y_: y_ + h, x_: x_ + w]
    resized_image = cv2.resize(roi, size)

    # prepare the image for prediction
    x = image.img_to_array(resized_image)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)

    # making prediction
    predicted_prob = model.predict(x)
    print(predicted_prob)
    print(predicted_prob[0].argmax())
    print("Predicted face: " + class_list[predicted_prob[0].argmax()])
    print("============================\n")


# In[33]:


import cv2
import os
import pickle
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

headshots_folder_name = 'Headshots'

# dimension of images
image_width = 224
image_height = 224

# for detecting faces
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# set the directory containing the images
images_dir = os.path.join(".", headshots_folder_name)

current_id = 0
label_ids = {}

# iterates through all the files in each subdirectories
for root, _, files in os.walk(images_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
        # path of the image
            path = os.path.join(root, file)

        # get the label name (name of the person)
        label = os.path.basename(root).replace(" ", ".").lower()

        # add the label (key) and its number (value)
        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1

        # load the image
            imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
            image_array = np.array(imgtest)

        # get the faces detected in the image
        faces =  facecascade.detectMultiScale(imgtest,
            scaleFactor=1.1, minNeighbors=5)

        # if not exactly 1 face is detected, skip this photo
        if len(faces) != 1:
            print(f'---Photo skipped---\n')
        # remove the original image
        os.remove(path)
        continue

        # save the detected face(s) and associate
        # them with the label
        for (x_, y_, w, h) in faces:

            # draw the face detected
            face_detect = cv2.rectangle(imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)
            plt.imshow(face_detect)
            plt.show()

            # resize the detected face to 224x224
            size = (image_width, image_height)

            # detected face region
            roi = image_array[y_: y_ + h, x_: x_ + w]

            # resize the detected head to target size
            resized_image = cv2.resize(roi, size)
            image_array = np.array(resized_image, "uint8")

            # remove the original image
            os.remove(path)

            # replace the image with only the face
            im = Image.fromarray(image_array)
            im.save(path)


# In[29]:


type(np.NaN)


# In[88]:


for (x_, y_, w, h) in faces:
    # draw the face detected
        face_detect = cv2.rectangle(imgtest, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)
        plt.imshow(face_detect)
        plt.show()

    # resize the detected face to 224x224
        size = (image_width, image_height)
        roi = image_array[y_: y_ + h, x_: x_ + w]
        resized_image = cv2.resize(roi, size)

    # prepare the image for prediction
        x = image.img_to_array(resized_image)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)

    # making prediction
        predicted_prob = model.predict(x)
        print(predicted_prob)
        print(predicted_prob[0].argmax())
        print("Predicted face: " + class_list[predicted_prob[0].argmax()])
        print("============================\n")


# In[34]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator =  train_datagen.flow_from_directory(
                './Headshots',
                target_size=(224,224),
                color_mode='rgb',
                batch_size=32,
                class_mode='categorical',
                shuffle=True)


# In[35]:


train_generator.class_indices.values()
# dict_values([0, 1, 2])
NO_CLASSES = len(train_generator.class_indices.values())


# In[36]:


get_ipython().system('pip install keras_vggface')


# In[37]:


from tensorflow.keras.layers import Layer, InputSpec


# In[38]:


from keras.utils import get_source_inputs
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.utils import layer_utils

import tensorflow as tf
import keras
import keras_vggface
from keras_vggface.vggface import VGGFace
import mtcnn
import numpy as np
import matplotlib as mpl
from keras.utils.data_utils import get_file
import keras_vggface.utils
import PIL
import os
import os.path


# In[39]:


vggface=VGGFace(model='vgg16')


# In[55]:


f#rom keras.models import model_from_json
# Weights: https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5
#model.load_weights('vgg_face_weights.h5')


# In[56]:


#vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


# In[40]:


from keras_vggface.vggface import VGGFace

model = VGGFace(model='vgg16')
# same as the following
model = VGGFace() # vgg16 as default


# In[41]:


model.summary()


# In[42]:


model = VGGFace(model='resnet50')


# In[13]:


#model.summary()


# In[43]:


model = VGGFace(model='senet50')


# In[11]:


#model.summary()


# In[44]:


import numpy as np
#from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import cv2
import keras.utils as image
# load the image
img = image.load_img(
    './Matthias_Sammer.jpg',
    target_size=(224, 224,3))

# prepare the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = utils.preprocess_input(x, version=1)

# perform prediction
preds = model.predict(x)
print('Predicted:', utils.decode_predictions(preds))


# In[28]:


plt.imshow(img)


# In[ ]:





# In[24]:


# load the photograph
pixels = cv2.imread('.Hearshorts/Tom/Tom1.webp')


# In[ ]:





# In[34]:


# plot photo with detected faces using opencv cascade classifier
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
# load the photograph
pixels = imread('C:/Users/pc/OneDrive/Pictures/PIC1.jpeg')
# load the pre-trained model
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detectionq
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
 # extract
    x, y, width, height = box
    x2, y2 = x + width, y + height
 # draw a rectangle over the pixels
    rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
# show the image
    cv2.imshow('face detection', pixels)
# keep the window open until we press a key
waitKey(0)
# close the window
destroyAllWindows()


# In[35]:


# confirm mtcnn was installed correctly
import mtcnn
# print version
print(mtcnn.__version__)


# In[36]:


model = MTCNN(weights_file='filename.npy')


# In[ ]:




