#seq2seq translation modeling for sum of several numbers as an string data generated dataset to predict sum as an string
#sample :  " 2+14+13+3" ---> " 32"


import numpy as np
import tensorflow as tf
from math import log10 , ceil

def create_pairs(n_sample , n_number , large):
  X , y = [] , []
  for i in range(n_sample):
    x_s = [np.random.randint(0,large) for j in range(n_number)]
    y_s = sum(x_s)
    X.append(x_s)
    y.append(y_s)
  return X , y

def num2str(X , y , n_number , largest):
  max_len = n_number * ceil(log10(largest + 1)) + n_number - 1
  X_str , y_str = [] , []
  for x_s in X:
    x_string = '+'.join([str(i) for i in x_s])
    x_string += ''.join([' ' for j in range(max_len - len(x_string))])
    X_str.append(x_string)
  
  max_len = ceil(log10(n_number * (largest+1)))
  for y_s in y:
    y_string = str(y_s)
    y_string += ''.join([' ' for j in range(max_len - len(y_string))])
    y_str.append(y_string)
  return X_str , y_str

def str2seq(X , y , chars):
  char2index = {v : i for i,v in enumerate(chars)}
  X_seq , y_seq = [] , []
  for x_s in X:
    seq_x= [char2index[i] for i in x_s]
    X_seq.append(seq_x)
  for y_s in y:
    seq_y= [char2index[i] for i in y_s]
    y_seq.append(seq_y)
  return X_seq , y_seq

def one_hot_encode(X , y , vocab_len):
  X_enc , y_enc = [] , []
  for x_s in X:
    enc_x = [[1 if j==i else 0 for j in range(vocab_len)] for i in x_s]
    X_enc.append(enc_x)
  for y_s in y:
    enc_y = [[1 if j==i else 0 for j in range(vocab_len)] for i in y_s]
    y_enc.append(enc_y)
  return np.array(X_enc) , np.array(y_enc)

def generate_data(n_sample , n_number , largest, chars):
  X , y = create_pairs(n_sample , n_number , largest)
  X , y = num2str(X , y , n_number , largest)
  X , y = str2seq(X , y , chars)
  X , y = one_hot_encode(X , y , len(chars))
  return X , y

n_sample = 10000
n_number = 5
largest = 85
chars = ['0','1','2','3','4','5','6','7','8','9',' ','+']
max_len_input = n_number * ceil(log10(largest + 1)) + n_number - 1
max_len_output = ceil(log10(n_number * (largest+1)))

X , y = generate_data(n_sample , n_number , largest, chars)


model = tf.keras.models.Sequential([
                                    tf.keras.layers.LSTM(256 , input_shape = (max_len_input, len(chars))),
                                    tf.keras.layers.RepeatVector(max_len_output),
                                    tf.keras.layers.LSTM(64 , return_sequences = True),
                                    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(chars), activation = 'softmax'))
])

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()

for i in range(20):
  X , y = generate_data(n_sample , n_number , largest, chars)
  model.fit(
    X , y,
    batch_size = 64,
    epochs = 10
  )