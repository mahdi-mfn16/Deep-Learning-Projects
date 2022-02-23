#input dat from kaggle 
#weather dataset
#prediction simple RNN model with LSTM layers to predict mode of weather at tomorrow

! pip install kaggle
! mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/kaggle.json
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download muthuj7/weather-dataset
! unzip weather-dataset
! rm weather-dataset.zip

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file):
  mms = MinMaxScaler()
  data = pd.read_csv(file)
  data['Pressure (millibars)'] = data['Pressure (millibars)'].apply(lambda x : x if x > 400 else np.nan)
  data = data.fillna(method="ffill")
  data = data.drop(['Daily Summary' , 'Loud Cover'] , axis = 1)
  summary = pd.get_dummies(data['Summary'])
  Precip_Type	= pd.get_dummies(data['Precip Type'])
  data = data.drop(['Precip Type' , 'Summary'] , axis = 1)
  data = pd.concat([data, Precip_Type , summary], axis=1)
  data.iloc[: ,1:] = mms.fit_transform(data.iloc[: ,1:])
  return data

def create_data(data , period , train_part):
  X , y = [] , []
  data = data.iloc[:,1:].values
  for i in range(len(data) - period):
    x_s = data[i:i + period,1:10]
    y_s = data[i + period , 10:]
    X.append(x_s)
    y.append(y_s)
  X , y = np.array(X) , np.array(y)
  train_end_index = int(len(X) * train_part)
  X_train , y_train , X_test , y_test = X[:train_end_index] , y[:train_end_index] , X[train_end_index:] , y[train_end_index:]
  return X_train , y_train , X_test , y_test

data = preprocess_data('weatherHistory.csv')
data.head()

X_train , y_train , X_test , y_test = create_data(data , 30 , 0.7)

model = tf.keras.models.Sequential([
                                    tf.keras.layers.LSTM(256 , return_sequences=True , input_shape = (30 , 9)),
                                    tf.keras.layers.LSTM(512 , return_sequences=True),
                                    tf.keras.layers.LSTM(256 , return_sequences=False),
                                    tf.keras.layers.Dense(128 , activation = 'relu'),
                                    tf.keras.layers.Dense(64 , activation = 'relu'),
                                    tf.keras.layers.Dense(26 , activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(
    X_train , y_train,
    validation_data = (X_test , y_test),
    batch_size = 64,
    epochs = 100
)