#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 23:26:58 2017

@author: syedrahman
"""

import os

os.chdir('/Users/syedrahman/Documents/Summer2017/Insight/Project')


import numpy as np
import sys, re
import itertools
from collections import Counter
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from matplotlib import pyplot

# CNN for the humor dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

data = pd.read_csv('humor_dataset.csv')  

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clear(string):
    try:
        sen = clean_str(string)       
        return sen
    except:
        return 'NC'

def postprocess(data, n=100000):
    data = data.sample(n)
    data['text1'] = data['text'].progress_map(clear)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.text1 != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

newdata = postprocess(data)

x = np.array(newdata.text1)
y = np.array(newdata.funny)

top_words = 5000

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = top_words) 

data_features = vectorizer.fit_transform(x)
data_features = data_features.toarray()

vocab = vectorizer.get_feature_names()
print(vocab)

print(data_features.shape)

type(data_features)

# sorting the data according to max column frequency 
b = np.sum(data_features, axis = 0)
idx = b.argsort()
data_features = data_features[:,idx]

X_train = data_features[0:80000,:]
X_test = data_features[80000:100000,:]
y_train = np.array(newdata.funny)[0:80000] 
y_test = np.array(newdata.funny)[80000:100000]

	
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

########################################################################
###    Simple Multi-Layer Perceptron Model                           ###
########################################################################
	
# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


##############################################################################
# One-Dimensional Convolutional Neural Network Model                         #
##############################################################################
# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary()) 

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('model.h5')         

prediction = model.predict(X_test[0])
print(prediction)