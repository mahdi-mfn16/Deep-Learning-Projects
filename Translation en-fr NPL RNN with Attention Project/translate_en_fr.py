#translation model on en-fr dataset from kaggle
#NLP RNN Model with Attention layer
#sub_classing model


#-------------------Linux Commands----------------------
! pip install kaggle
! mkdir ~/.kaggle
! cp /content/drive/MyDrive/kaggle.json ~/.kaggle/kaggle.json
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download dhruvildave/en-fr-translation-dataset
! unzip en-fr-translation-dataset
! rm en-fr-translation-dataset.zip
#-------------------------------------------------------

import pandas as pd
import numpy as np
import unicodedata
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

def preprocess_txt(txt):
  txt = txt.lower().strip()
  txt = ''.join([t for t in unicodedata.normalize('NFD' , txt) if unicodedata.category(t) != 'Mn'])
  txt = re.sub(r"([?.~,])" , r" \1" , txt)
  txt = re.sub(r"[' ']+" , " " , txt)
  txt = re.sub(r"[^a-zA-Z0-9?.~,]+" , " " , txt)
  txt = txt.rstrip().strip()
  txt = '<start> ' + txt + ' <end>'
  return txt

def max_len(data):
  maxlen = max([len(d) for d in data])
  return maxlen

input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
output_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

def tokenize_data(txt_data,tokenizer):
  tokenizer.fit_on_texts(txt_data)
  txt_seq = tokenizer.texts_to_sequences(txt_data)
  txt_padded = tf.keras.preprocessing.sequence.pad_sequences(txt_seq , padding = 'post')
  return txt_padded

def load_dataset(n_skip , n_sample):
  data = pd.read_csv('en-fr.csv',names=['en' , 'fr'], skiprows=n_skip, nrows=n_sample).dropna()
  inputs = data.iloc[: , 0].apply(preprocess_txt)
  outputs = data.iloc[: , 1].apply(preprocess_txt)
  input_tensor = tokenize_data(inputs , input_tokenizer)
  output_tensor = tokenize_data(outputs , output_tokenizer)
  return input_tensor , output_tensor

input_max_length , output_max_length = [] , []
for i in range(100):
  input_tensor , output_tensor = load_dataset(n_skip=(i+1)*1000 , n_sample=1000)
  inp_maxlen = max_len(input_tensor)
  out_maxlen = max_len(output_tensor)
  input_max_length.append(inp_maxlen)
  output_max_length.append(out_maxlen)

batch_size = 16
embed_dim = 256
gru_units = 1024
vocab_input_size = len(input_tokenizer.word_index)+1
vocab_output_size = len(output_tokenizer.word_index)+1

class Encoder(tf.keras.Model):

  def __init__(self, vocab_size , embed_dim , batch_size , gru_units):
    super(Encoder , self).__init__()
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.batch_size = batch_size
    self.gru_units = gru_units
    self.embedding = tf.keras.layers.Embedding(self.vocab_size , self.embed_dim)
    self.gru = tf.keras.layers.GRU(self.gru_units , return_sequences = True , return_state = True)

  def call(self , x , hidden):
    x = self.embedding(x)
    output , state = self.gru(x , initial_state = hidden)
    return output , state

  def initial_hidden_state(self):
    return tf.zeros((self.batch_size , self.gru_units))

encoder = Encoder(vocab_size = vocab_input_size , embed_dim = embed_dim , batch_size = batch_size , gru_units = gru_units)


class Attention(tf.keras.layers.Layer):
  def __init__(self , dense_units):
    super(Attention , self).__init__()
    self.dense_units = dense_units
    self.w1 = tf.keras.layers.Dense(self.dense_units)
    self.w2 = tf.keras.layers.Dense(self.dense_units)
    self.v = tf.keras.layers.Dense(1)

  def call(self , states , outputs):
    same_dim_states = tf.expand_dims(states , axis = 1)
    score = self.v(tf.nn.tanh(self.w1(outputs) + self.w2(same_dim_states)))
    attention_weight = tf.nn.softmax(score , axis = 1)
    attention_vector = attention_weight * outputs
    attention_vector = tf.reduce_sum(attention_vector , axis = 1)
    return attention_vector , attention_weight



class Decoder(tf.keras.Model):
  def __init__(self , vocab_size , embed_dim , batch_size , gru_units):
    super(Decoder , self).__init__()
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.batch_size = batch_size
    self.gru_units = gru_units
    self.embedding = tf.keras.layers.Embedding(self.vocab_size , self.embed_dim)
    self.gru = tf.keras.layers.GRU(self.gru_units , return_sequences = True , return_state = True)
    self.fc = tf.keras.layers.Dense(self.vocab_size , activation = 'softmax')
    self.attention = Attention(self.gru_units)

  def call(self , targ_x , hidden , enc_output):
    attention_vectors , attention_weights = self.attention(hidden , enc_output)
    targ_x = self.embedding(targ_x)
    targ_x = tf.concat([tf.expand_dims(attention_vectors , 1) , targ_x], axis = -1)
    outputs , states = self.gru(targ_x)
    outputs = tf.reshape(outputs , (-1 , outputs.shape[2]))
    prediction = self.fc(outputs)
    return prediction , states , attention_weights

decoder = Decoder(vocab_size = vocab_output_size , embed_dim = embed_dim , batch_size = batch_size , gru_units = gru_units)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True , reduction = 'none')
checkpoint = tf.train.Checkpoint(optimizer = optimizer , encoder = encoder , decoder = decoder)

def loss_function(real , pred):
  mask = tf.math.logical_not(tf.math.equal(real , 0))
  loss = loss_object(real , pred)
  mask = tf.cast(mask , dtype = loss.dtype)
  loss = loss* mask
  loss = tf.reduce_mean(loss)
  return loss

def train_step(inputs , targets , hiddien_state):
  encoder_hiddien_state = hiddien_state
  decoder_hiddien_state = encoder_hiddien_state
  loss = 0
  with tf.GradientTape() as tape:
      enc_outputs , encoder_hiddien_state = encoder(inputs , encoder_hiddien_state)
      dec_input = tf.expand_dims([input_tokenizer.word_index['<start>']] * batch_size , 1)    
      for t in range(1 , targets.shape[1]):
          prediction , decoder_hiddien_state , attention_weights = decoder(targ_x = dec_input , hidden = decoder_hiddien_state , enc_output = enc_outputs)
          loss += loss_function(targets[: , t] , prediction)
          dec_input = tf.expand_dims(targets[: , t] , 1)
  batch_loss = loss/int(targets.shape[1])
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss , variables)
  optimizer.apply_gradients(zip(gradients , variables))
  return batch_loss


for j in range(100):
  input_tensor , output_tensor = load_dataset(n_skip=j*1000 , n_sample=1000)
  dataset = tf.data.Dataset.from_tensor_slices((input_tensor , output_tensor)).batch(batch_size , drop_remainder = True)
  epochs = 100
  for i in range(epochs):
    encoder_hiddien_state = encoder.initial_hidden_state()
    total_loss = 0
    print('Epoch :' , str(i+1))
    for batch , (inputs , targets) in enumerate(dataset.take(len(dataset))):
      batch_loss = train_step(inputs , targets , encoder_hiddien_state)
      total_loss += batch_loss
      print('Batch', str(batch + 1) ,'_loss:' , batch_loss.numpy()) 
    print('total_loss: ' , total_loss.numpy())
    checkpoint.save(file_prefix='checkpoint/model')


def translate(sen):
  sen = preprocess_txt(sen)
  seq = tf.expand_dims([input_tokenizer.word_index[i] for i in sen.split(' ')] , 0)
  seq = tf.keras.preprocessing.sequence.pad_sequences(seq , maxlen = inp_maxlen , padding = 'post')
  encoder_hiddien_state = tf.expand_dims(tf.zeros(gru_units) , 0)
  dec_input = tf.expand_dims([output_tokenizer.word_index['<start>']],0)
  enc_outputs , encoder_hiddien_state = encoder(seq , encoder_hiddien_state)
  decoder_hiddien_state = encoder_hiddien_state
  result = '<start>'
  for t in range(out_maxlen):
          prediction , decoder_hiddien_state , attention_weights = decoder(targ_x = dec_input , hidden = decoder_hiddien_state , enc_output = enc_outputs)
          predict_index = tf.argmax(prediction[0]).numpy()
          dec_input = tf.expand_dims([predict_index],0)
          result += ' ' + output_tokenizer.index_word[predict_index]
          if predict_index == output_tokenizer.word_index['<end>']:
            return result , sen
  return result , sen

checkpoint.restore(tf.train.latest_checkpoint('checkpoint'))

translate('you are so beautiful')