#classification of fruit images with CNN 
#dataset from kaggle
#97% accuracy for validation dataset

#-----------------------------------------------------------
! pip install kaggle
! mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/kaggle.json
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download kritikseth/fruit-and-vegetable-image-recognition
! unzip fruit-and-vegetable-image-recognition
! rm fruit-and-vegetable-image-recognition.zip
#-----------------------------------------------------------


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv2D , Dropout , MaxPool2D , Flatten , BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3


model = InceptionV3(weights = None, input_shape = (299,299,3), classes = 36)
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

mch = ModelCheckpoint('model/model_fruits.h5' , monitor = 'val_loss' , mode = 'min')

model = load_model('model/model_fruits.h5')
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

mch = ModelCheckpoint('model/model_fruits.h5' , monitor = 'val_loss' , mode = 'min')

gen = ImageDataGenerator(
    rescale = 1./255.,
    # rotation_range = 40,
    # zoom_range = 
)

train = gen.flow_from_directory(
    'train',
    color_mode='rgb',
    target_size = (299,299),
    class_mode = 'categorical',
    batch_size = 10
)

test = gen.flow_from_directory(
    'test',
    color_mode='rgb',
    target_size = (299,299),
    class_mode = 'categorical',
    batch_size = 10
)

train.class_indices

model = Sequential([
                    Conv2D(32,3,input_shape = (256,256,3) , activation = 'relu'),
                    BatchNormalization(),
                    MaxPool2D(pool_size = (2,2) , strides=2),
                    Conv2D(64,3, activation = 'relu'),
                    MaxPool2D(pool_size = (2,2) , strides=2),
                    Conv2D(128,3, activation = 'relu'),
                    MaxPool2D(pool_size = (2,2) , strides=2),
                    Conv2D(128,3, activation = 'relu'),
                    MaxPool2D(pool_size = (2,2) , strides=2),
                    Conv2D(64,3, activation = 'relu'),
                    MaxPool2D(pool_size = (2,2) , strides=2),
                    Conv2D(32,3, activation = 'relu'),
                    MaxPool2D(pool_size = (2,2) , strides=2),
                    Flatten(),
                    Dense(64 , activation = 'relu'),
                    Dense(128 , activation = 'relu'),
                    Dense(36 , activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

mch = ModelCheckpoint('/content/drive/MyDrive/datasets/model_fruits.h5' , monitor = 'val_loss' , mode = 'min')


hist = model.fit(
    train,
    validation_data = test,
    steps_per_epoch = len(train),
    validation_steps = len(test),
    epochs = 30,
    
    callbacks = [mch]
)

valid = gen.flow_from_directory(
    'validation',
    color_mode='rgb',
    target_size = (299,299),
    class_mode = 'categorical',
    batch_size = 10
)

pred = model.predict(valid)

plt.imshow(plt.imread('validation/banana/Image_1.jpg')[::-1])

import cv2
img = cv2.imread('train/apple/Image_20.jpg')
img = cv2.resize(img, (299,299))
img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
x = np.expand_dims(img,axis = 0)
plt.imshow(img)

np.argmax(model.predict(x))

x.shape

train.class_indices

np.argmax(pred[0])

import matplotlib.pyplot as plt
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['accuracy'])