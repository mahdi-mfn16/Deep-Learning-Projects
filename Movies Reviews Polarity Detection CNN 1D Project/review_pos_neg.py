#Polarity detection for big dataset of reviews of movies
#1D CNN model

! unrar x reviewdata.rar
! mkdir bigreview
! unrar x bigreview.rar bigreview

import numpy as np
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
sw = nltk.corpus.stopwords.words('english')

with open('stopwords_eng.txt' , 'r') as f:
  sw2 = f.readlines()
  sw2 = [i.replace('\n' ,'') for i in sw2]
sw = list(set(sw + sw2))

def make_data(path):
  X , y , lens = [] , [] , []
  for c in os.listdir(path):
    for p in os.listdir(os.path.join(path,c)):
      with open(os.path.join(os.path.join(path,c),p), 'r') as f:
        text = f.read()
        text = text.lower()
        words = nltk.word_tokenize(text)
        words= [w for w in words if w.isalnum()]
        f_words= [w for w in words if not w in sw]
        text = ' '.join(f_words)
        lens.append(len(text))
        X.append(text)
        y.append(c)
  return np.array(X) , np.array(y) , max(lens)

X_train , y_train , maxlen_train = make_data('bigreview/train')
X_test , y_test , maxlen_test = make_data('bigreview/test')
X = np.append(X_train, X_test, axis=0)
maxlens = max(maxlen_train , maxlen_test)

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
le = LabelEncoder()
y_train , y_test = to_categorical(le.fit_transform(y_train)) , to_categorical(le.fit_transform(y_test))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten , Conv1D , MaxPool1D , Embedding , Dense , BatchNormalization , Dropout , Bidirectional , LSTM

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
seq_train = tokenizer.texts_to_sequences(X_train)
seq_test = tokenizer.texts_to_sequences(X_test)
pad_train = pad_sequences(seq_train , maxlen = maxlens , padding = 'post')
pad_test = pad_sequences(seq_test , maxlen = maxlens , padding = 'post')

model = Sequential([
                    Embedding(input_dim = len(tokenizer.index_word) + 1 , output_dim = 60 , input_shape = (maxlens,)),         
                    Conv1D(filters = 64 , kernel_size = 4 , activation = 'relu'),
                    Dropout(0.6),
                    Conv1D(filters = 64 , kernel_size = 3 , activation = 'relu'),
                    Dropout(0.6),
                      
                    MaxPool1D(pool_size = 3),
                    BatchNormalization(),
                    
                    Conv1D(filters = 32 , kernel_size = 4 , activation = 'relu'),
                    Dropout(0.6),
                    Conv1D(filters = 32 , kernel_size = 3 , activation = 'relu'),
                    
                    MaxPool1D(pool_size = 2),
                    BatchNormalization(),
                    Dropout(0.6),
                    # Bidirectional(LSTM(units = 128, return_sequences=True)),
                    # Dropout(0.6),
                    
                    Flatten(),
                    Dense(units = 64 , activation = 'relu'),
                    BatchNormalization(),
                    Dropout(0.5),
                    Dense(units = 32 , activation = 'relu'),
                    BatchNormalization(),
                    Dropout(0.5),
                    Dense(units = 16 , activation = 'relu'),
                    BatchNormalization(),
                    Dropout(0.4),
                    Dense(units = 2 , activation = 'softmax')

])

model.compile(
    optimizer = 'Adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

hist = model.fit(
    pad_train,
    y_train,
    validation_data = (pad_test , y_test),
    epochs = 10,
    batch_size = 10
)