#modules for building LSTM model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import keras.utils as ku

from tensorflow.random import set_seed
from numpy.random import seed
set_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

data = '/Users/spencerchapman/Google Drive/My Drive/CSU_GLOBAL/CSC525/Module_8/archive/twcs/twcs.csv'
df = pd.read_csv(data)
records = int(len(df)/300)
print(records)
text = df['text'][:records]

#data cleaning
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    return txt

corpus = [clean_text(x) for x in text]
# corpus[:10]

#generating sequence of n-gram tokens
tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    #tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    #convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
inp_sequences[:10]

#padding the sequences to obtain Variables: Predictors and Target

def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding = 'pre'))
    predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
#now we are able to obtain the input vector X and the label vector Y which can be used for the training purposes.

#creating the LSTM model
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    #Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    #Add hidden layer 1 - LSTM layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    model.add(Dense(total_words, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = create_model(max_sequence_len, total_words)
model.summary()

history = model.fit(predictors, label, epochs=5, verbose=5)
acc = history.history['loss']

import matplotlib.pyplot as plt
plt.plot(acc, label = 'Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
# evaluate the model on the test set

def generate_text(seed_text, model, tokenizer, max_sequence_len):
    for _ in range(50):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
                                   maxlen=max_sequence_len-1, 
                                   padding='pre')
        predicted = model.predict(token_list, verbose=0)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(predicted):
                output_word = word
                break
        seed_text += " " + output_word
        
    return seed_text.title()


seed_text = "Hello"
generated_text = generate_text(seed_text, model, tokenizer, max_sequence_len)
print(generated_text)
