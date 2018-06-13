import numpy as np
import re
import string 
import pickle
from unicodedata import normalize
import matplotlib.pyplot as plt
import argparse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
#from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras_attention import AttentionDecoder
from keras.layers import Dropout
import keras
import tensorflow as tf
from keras import backend as K

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
    #    re_print = re.compile("[^{}]".format(re.escape(string.printable)))  #add escape seq(backslash) in front of everything except alphanumeric and _
    punc = string.punctuation + "„" + "–"+"’"+'“'     
    table = str.maketrans("","",punc)                     # mapping punctuations to None, hence, removing them
    for pair in lines:
        clean_pair=[]
        for line in pair:
            # handling normalization of unicode chars
    #            line = normalize("NFD",line).encode('ascii','ignore')
    #            line = line.decode("UTF-8")
            # tokenize on space
            line = line.split()
            # to lowercase
    #            line = [word.lower() for word in line]
            # remove punctuation here
            line = [word.translate(table) for word in line]
            line = [word.lower() for word in line]
            # remiving non-printable chars
    #            line = [re_print.sub('',w) for w in line]
            #remove digits
    #            line = [word for word in line if word.isalpha()]
            # appending as string
            clean_pair.append(" ".join(line))
        cleaned.append(clean_pair)
    return np.array(cleaned)
    
def save_clean_data(sentences, filename):
    pickle.dump(sentences, open(filename,'wb'))
    print("saved: {}".format(filename))


def load_clean_dataset(filename):
    return np.load(open(filename, "rb"))
    
#K.set_floatx("float16")
 
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

def new_gen(trainX, trainY, batch_size, vocab_size):
    X = np.zeros((batch_size, trainX.shape[1]))
    Y = np.zeros((batch_size, trainY.shape[1], vocab_size))
    while (True):
        #        print("This is i {}".format(i))
        for b in range(0,len(trainX),batch_size):
            ylist=list()
            if(b+batch_size>=len(trainX)):
                break
            X = trainX[b:b+batch_size]
            for i in range(b,b+batch_size):
                encoded = to_categorical(trainY[i],num_classes=vocab_size) 
                ylist.append(encoded)
                Y=np.array(ylist)
                Y= Y.reshape(batch_size, trainY.shape[1], vocab_size)
            yield X,Y
            
    
def read_all_embeddings(filename):
    embeddings = dict()
    with open(filename) as f:
        for line in f:
            line = line.split()
            if len(line)==65:            
                word = line[0]
                coeffs = line[1:]
                embeddings[word]= coeffs
    return embeddings
            

# bidirectional Encoder without attention
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
#    gpu_opts = tf.GPUoptions(per_process_gpu_memory_fraction = 0.5)   ### handling the gpu allocator error
#    sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_opts))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    K.set_session(sess)
    
    model = Sequential()
    # encoder =>
    model.add(Embedding(src_vocab,embedding_mat.shape[1] , input_length=src_timesteps, mask_zero=True, weights=[embedding_mat], trainable=False)) #check its output and shape removed n_units
    model.add(Dropout(0.45))
    model.add(Bidirectional(LSTM(n_units,return_sequences=True))) # uses previous output as input and performs computation on it.
    model.add(Dropout(0.45))
    model.add(Bidirectional(LSTM(n_units))) # tweak n_units
    model.add(RepeatVector(tar_timesteps)) 
    #decoder =>           
#    model.add(AttentionDecoder(int(n_units/8),tar_vocab))
    model.add(LSTM(n_units,return_sequences=True)) # +1 LSTM layer for decoding
    model.add(LSTM(n_units,return_sequences=True))
    model.add(LSTM(n_units,return_sequences=True))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    
    return model


def mat_builder(embed,tokenizer):#ger_tokenizer
    embedding_mat = np.zeros((ger_vocab_size, 64))
    count=0
    for word,i in tokenizer.word_index.items():
        embedding_vec = embed.get(word)
        if embedding_vec is not None:
            embedding_mat[i]= embedding_vec
        else:
            count+=1
    return embedding_mat,count
        
    
class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self,logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.accuracies.append(logs.get("acc"))

#==============================================================================
# print(min(loss_acc_hist.losses))
# plt.plot(loss_acc_hist.accuracies)
# plt.title("Batchwise accuracies")
# plt.xlabel("batches")
# plt.ylabel("Accuracy")
# plt.show()
# 
# plt.plot(history.history["acc"])
# plt.plot(history.history["val_acc"])
# plt.xlabel("epochs")
# plt.title("Acc per epoch")
# plt.ylabel("accuracy")
# plt.legend(["train","test/val"])
# plt.show()
# 
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.legend(["train","test"])
# plt.title("Losses on training and test")
# plt.ylabel("Loss")
# plt.xlabel("epoch")
# plt.show()
# print(max(history.accuracies))
#                 
#==============================================================================
