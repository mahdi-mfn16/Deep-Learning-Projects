#Classification of chest Xray Images Datasets from kaggle

#-------------------linux commands-----------------------------
! pip install kaggle
! mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/kaggle.json
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download paultimothymooney/chest-xray-pneumonia
! unzip chest-xray-pneumonia
! rm chest-xray-pneumonia.zip
! rm -rf chest_xray/__MACOSX
! rm -rf chest_xray/chest_xray
! pip install mahotas --upgrade --ignore-installed
! pip install catboost

#---------------------------------------------------------------

import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
import os
import pickle
from mahotas import features
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier



#create datasets and preprocessing

def img_process(img):
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  imge = cv2.resize(img,(1000,1000))
  # x0 = imge.flatten()
  x1 = cv2.calcHist(img , [0] , None , [256] , [0,256]).flatten()
  x2 = cv2.HuMoments(cv2.moments(img)).flatten()
  x3 = features.haralick(img).mean(axis=0) 
  feature = np.hstack([x1,x2,x3])
  return feature

def make_data(path):
  X = []
  y = []
  paths = os.path.join('chest_xray',path)
  for label in os.listdir(paths):
    curr_dir = os.path.join(paths,label)
    imgs = os.listdir(curr_dir)
    print('directory: ', path, label)
    for im in imgs:
      print('image:', im)
      img = os.path.join(curr_dir , im)
      image = cv2.imread(img)
      feature = img_process(img)
      target = label
      X.append(feature)
      y.append(target)
  X = np.array(X)
  y = np.array(y)
  return X , y

X_train , y_train = make_data('train')
X_test , y_test = make_data('test')

mms = MinMaxScaler()
X_train , X_test = mms.fit_transform(X_train) , mms.fit_transform(X_test)
le = LabelEncoder()
y_train , y_test = le.fit_transform(y_train) , le.fit_transform(y_test)


#models and prediction


# XGBoost Classification
xgb = XGBClassifier(n_estimators = 1000, random_state = 1000)
xgb.fit(X_train , y_train)
y_pred = xgb.predict(X_test)
print(metrics.classification_report(y_true = y_test , y_pred = y_pred))


# CatBoost Classification
cat = CatBoostClassifier(n_estimators=1000 , random_state = 1000 , eval_metric='Accuracy')
cat.fit(X_train ,y_train, eval_set = (X_test , y_test))
y_pred = cat.predict(X_test)
print(metrics.classification_report(y_true = y_test , y_pred = y_pred))



#NN Classification
y_train_nn , y_test_nn = tf.keras.utils.to_categorical(y_train) , tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential([
                                    
                                    tf.keras.layers.Dense(276,activation = 'relu'),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Dense(200,activation = 'relu'),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Dense(120,activation = 'relu'),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Dense(320,activation = 'relu'),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Dense(120,activation = 'relu'),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Dense(120,activation = 'relu'),
                                    tf.keras.layers.Dense(2,activation = 'softmax'),
])

model.compile(
    optimizer = tf.optimizers.Adam(learning_rate = 0.001),
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
    metrics = [tf.keras.metrics.CategoricalAccuracy()]
)

history = model.fit(
    x = X_train,
    y = y_train_nn,
    batch_size = 10,
    epochs = 100,
    validation_data = (X_test, y_test_nn)
)

plt.plot(history.history['categorical_accuracy'],c = 'green')
plt.plot(history.history['val_categorical_accuracy'],c = 'blue')