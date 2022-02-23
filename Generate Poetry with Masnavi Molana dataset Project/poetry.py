#generate poetry from web scrapted masnavi dataset 
#model many2many RNN
#generate character by character to generata poem

import numpy as np
import tensorflow as tf
from IPython.display import clear_output 
import datetime as dt

data = open('masnavi','r', encoding= 'UTF-8').read()

data = data.replace('\xa0' , ' ')
data = data.replace('\u200c' , ' ')
data = data.replace('(' , '')
data = data.replace(')' , '')
data = data.replace('«' , '')
data = data.replace('»' , '')
data = data.replace('؛' , '')
data = data.replace('،' , '')
data = data.replace('؟' , '')
data = data.replace('ء' , '')
data = data.replace('أ' , 'ا')
data = data.replace('آ','ا')
data = data.replace('ؤ' , 'و')
data = data.replace('ئ' , 'ی')
data = data.replace('َ' , '')
data = data.replace( 'ُ' , '')
data = data.replace('ِ' , '')
data = data.replace('ّ' , '')
data = data.replace('ْ' , '')
data = data.replace('ٔ' , '')
data = data.replace('*' , '')
data = data.replace('-' , ' ')
data = data.replace('[' , '')
data = data.replace(']' , '')
data = data.replace('ً' , '')
data = data.replace('ٌ' , '')
data = data.replace( 'ٍ' , '')
data = data.replace( 'ٖ' , '')
data = data.replace( 'ٰ' , '')
data = data.replace('\u200d' , ' ')
data = data.replace('.' , '')


vocabs = sorted(set(data))
char2index = {c:i for i,c in enumerate(vocabs)}
index2char = np.array(vocabs)


model = tf.keras.models.Sequential([
                                    tf.keras.layers.Embedding(len(vocabs) , 512),
                                    tf.keras.layers.LSTM(2048 , return_sequences=True),
                                    # tf.keras.layers.Dropout(0.2),                                  
                                    tf.keras.layers.Dense(len(vocabs), activation = 'softmax')                                 
])

model.compile(
    optimizer = 'Adam', #tf.keras.optimizers.SGD(0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
)

mch = tf.keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/data/poetry/model/model1' , save_weights_only = True)
model.summary()


batchs = 17
lens = len(data)
ext = (lens//batchs) * batchs

def make_data(batch):  #for mapping tf dataset
  inp = batch[:-1]
  out = batch[1:]
  return inp , out

for i in range(batchs):    # each step generate new characters sequence from mesra :)) of poems 
  data = data[1:] + data[0:1]  #for rotation of all text to generate full sequence dataset characters
  d = np.array([char2index[c] for c in data[:ext]])
  d = tf.data.Dataset.from_tensor_slices(d)
  dataset = d.batch(17 , drop_remainder = True)
  dataset = dataset.map(make_data)
  dataset = dataset.batch(128 , drop_remainder = True)
   
  hist = model.fit(
    dataset,
    callbacks = [mch],
    epochs = 100
  )

model2 = tf.keras.models.Sequential([
                                    tf.keras.layers.Embedding(len(vocabs) , 512),
                                    tf.keras.layers.LSTM(2048 , return_sequences=True),
                                    # tf.keras.layers.Dropout(0.2),                                  
                                    tf.keras.layers.Dense(len(vocabs), activation = 'softmax')                                 
])

model2.build(tf.TensorShape([1,]))
model2.load_weights(tf.train.latest_checkpoint('/content/drive/MyDrive/data/poetry/model'))
model2.reset_states()

text = 'بس مبارک باد این فرخنده روز'

data_1 = tf.expand_dims([char2index[c] for c in text], 0 )

t = dt.datetime.now()
text_generated = index2char[data_1.numpy()[0]]
while True:
    delta = dt.datetime.now()-t
    if delta.microseconds >= 100000:
        
        pred = model2.predict(data_1)
        ids = np.argmax(pred[0], axis = 1)
        last_id = ids[-1]
        data_1 = tf.expand_dims(np.append(data_1.numpy()[0] , last_id)[1:], 0)
        text_generated = np.append(text_generated , index2char[last_id])
        clear_output(wait=True)
        print(''.join(text_generated))
        t = dt.datetime.now()