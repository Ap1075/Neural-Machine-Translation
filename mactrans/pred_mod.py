from keras.models import load_model
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#import tensorflow as tf
import string
#from keras.preprocessing.text import Tokenizer

#with tf.device("/gpu:0"):
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X
    
def word_from_id(integer, tokenizer):
    for word, i in tokenizer.word_index.items():
        if integer==i:
            return word
    return None

def predictions(tokenizer, source, model):
    prediction = model.predict(source,verbose=0)[0]
    indices = [argmax(vector) for vector in prediction]
    output =[]    
    for i in indices:
        word = word_from_id(i,tokenizer)
        if word is None:
            break
        output.append(word)
    return " ".join(output) 

def fin_pred_model(sources,tokenizer,model):
    for i,source in enumerate(sources):
#        source = np.array(source)
        source = source.reshape((1,source.shape[0]))
        translation = predictions(tokenizer, source, model)
    print (translation)

def punc_remover_lower(line):
    punc_rem = str.maketrans("","",string.punctuation)
#    for line in lines:
    line=line.translate(punc_rem)
    line = line.lower()
    return line
