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
    file = open(filename, mode="rt", encoding="utf-8")
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
filename = "deunew.txt"
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
no_sentences = 50000
dataset = raw_dataset[:no_sentences,:]
np.random.shuffle(dataset)
train, test = dataset[:45000],dataset[45000:]

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

def gen(trainX, trainY, batch_size, vocab_size):
    X = np.zeros((trainX.shape[0], trainX.shape[1]))
    Ylist = []
    i =0
    while True:
        for b in range(batch_size):
            if i == len(trainX):
                 i = 0
                 np.random.shuffle(trainX)
            X[i] = trainX[i]
            encoded = to_categorical(trainY[i], num_classes = vocab_size)
            i+=1       
            Ylist.append(encoded) 
        Y = np.array(Ylist)
        Ylist=[]
        Y = np.reshape(batch_size, trainY.shape[1], vocab_size)
        yield X, Y
            
    

# bidirectional Encoder without attention
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    # encoder =>
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True)) #check its output and shape
    model.add(Bidirectional(LSTM(n_units,return_sequences=True))) # uses previous output as input and performs computation on it.
    model.add(Bidirectional(LSTM(n_units))) # tweak n_units
    model.add(RepeatVector(tar_timesteps)) 
    #decoder =>
    model.add(LSTM(n_units,return_sequences=True)) # +1 LSTM layer for decoding
    model.add(LSTM(n_units,return_sequences=True))
    model.add(LSTM(n_units,return_sequences=True))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model
 
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: {}'.format(eng_vocab_size))
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))

# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)

# define model
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 512) # vocab size is total number of words, length is max sentence length
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# summarize defined model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

#fitting the model
filename = "4Input_4output_50000samples_rmsprop.h5"
checkpt = ModelCheckpoint(filename, monitor = "val_acc", verbose = 1, save_best_only = True, mode="max")
model.fit(trainX, trainY,  epochs= 75, batch_size = 128, validation_data = (testX,testY), callbacks = [checkpt], verbose = 2)

# Prediction
from keras.models import load_model
from numpy import argmax

model = load_model("4Input_4output_50000samples.h5")

# converting integer to word from tokenizer
def word_from_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generating target from source
def predict_sequence(model, tokenizer, source ):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = []
    for i in integers:
        word = word_from_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append(raw_target.split())
		predicted.append(translation.split())

trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])

evaluate_model(model, eng_tokenizer, trainX, train)
