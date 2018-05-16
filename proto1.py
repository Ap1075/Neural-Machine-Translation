# -*- coding: utf-8 -*-
"""
Created on Tue May 15 00:48:18 2018

@author: Armaan Puri 
"""
import pandas as pd
import numpy as np
import re
import string 
import pickle
from unicodedata import normalize
from sklearn.model_selection import train_test_split
# Data preprocessing for normalization

#loading the data
def load_doc(filename):
    file = open(filename, 'r', encoding="utf-8")
    text = file.read()
    file.close()
    return text

#separate eng and ger
def to_pairs(file):
    lines = file.strip().split('\n')
    pairs = [line.split('\t') for line in lines] 
    return pairs

# normalizing the pairs
def norm_pairs(lines):
    cleaned=[]
    re_print = re.compile("[^{}]".format(re.escape(string.printable)))  #add escape seq(backslash) in front of everything except alphanumeric and _
    table = str.maketrans("","",string.punctuation)                     # mapping punctuations to None, hence, removing them
    for pair in lines:
        clean_pair=[]
        for line in pair:
            # handling normalization of unicode chars
            line = normalize("NFD",line).encode('ascii','ignore')
            line = line.decode("UTF-8")
            # tokenize on space
            line = line.split()
            # to lowercase
            line = [word.lower() for word in line]
            # remove punctuation here
            line = [word.translate(table) for word in line]
            # remiving non-printable chars
            line = [re_print.sub('',w) for w in line]
            #remove digits
            line = [word for word in line if word.isalpha()]
            # appending as string
            clean_pair.append(" ".join(line))
        cleaned.append(clean_pair)
    return np.array(cleaned)
 
def save_clean_data(sentences, filename):
    pickle.dump(sentences, open(filename,'wb'))
    print("saved: {}".format(filename))
    
# input the dataset
filename = "deu.txt"
doc = load_doc(filename)
#splitting by languages
pairs = to_pairs(doc)
# cleaning
cleaned_pairs = norm_pairs(pairs)
# saving the normalized pairs
save_clean_data(cleaned_pairs,'english-german.pkl')

for i in range(100):
    print("{} => {}".format(cleaned_pairs[i,0],cleaned_pairs[i,1]))
    
def load_clean_dataset(filename):
    return np.load(open(filename, "rb"))

#load dataset
raw_dataset = load_clean_dataset("english-german.pkl")

############ REDUCING SIZE
no_sentences = 10000
dataset = raw_dataset[:no_sentences,:]
np.random.shuffle(dataset)
train, test = dataset[:8000],dataset[8000:]

#final save
save_clean_data(dataset, "english-german.pkl")
save_clean_data(train,'english-german-train.pkl')
save_clean_data(test,'english-german-test.pkl')

# loading the saved datasets
dataset = load_clean_dataset('english-german.pkl')
train = load_clean_dataset("english-german-train.pkl")
test = load_clean_dataset("english-german-test.pkl")

from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import RepeatVector
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
 
# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))
 
# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
 
# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)
 
# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X
 
# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
		 ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

# bidirectional Encoder without attention
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    # encoder =>
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(Bidirectional(LSTM(n_units)))
    model.add(Bidirectional(LSTM(n_units*2)))
    model.add(RepeatVector(tar_timesteps))
    #decoder =>
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model
 


# Inference