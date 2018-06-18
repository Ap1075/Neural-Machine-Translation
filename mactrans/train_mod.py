import numpy as np
#import re
import string 
import pickle
#from unicodedata import normalize
#import matplotlib.pyplot as plt
#import argparse
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
#from keras_attention import AttentionDecoder
from keras.layers import Dropout
#import keras
import tensorflow as tf
from keras import backend as K

class trainer(object):

    def __init__(self,model_path,word_vec_path,dataset_path):
        self.model_path = model_path
        self.word_vec_path = word_vec_path
        self.dataset_path=dataset_path
                
    def load_doc(self):
        file = open(self.dataset_path, mode="rt", encoding="utf-8")
        text = file.read()
        file.close()
        return text

#separate eng and ger
    def to_pairs(self,file):
        lines = file.strip().split('\n')
        pairs = [line.split('\t') for line in lines] 
        return pairs
    
    # normalizing the pairs
    def norm_pairs(self,lines):
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
        
    def save_clean_data(self,sentences, filename):
        pickle.dump(sentences, open(filename,'wb'))
        print("saved: {}".format(filename))
    
    
    def load_clean_dataset(self,filename):
        return np.load(open(filename, "rb"))
        
    #K.set_floatx("float16")
     
    # fit a tokenizer
    def create_tokenizer(self,lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer
     
    # max sentence length
    def max_length(self,lines):
        return max(len(line.split()) for line in lines)
     
    # encode and pad sequences
    def encode_sequences(self,tokenizer, length, lines):
    	# integer encode sequences
        X = tokenizer.texts_to_sequences(lines)
        # pad sequences with 0 values
        X = pad_sequences(X, maxlen=length, padding='post')
        return X
     
    # one hot encode target sequence using gen
    def new_gen(self,trainX, trainY, batch_size, vocab_size):
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
                
        
    def read_all_embeddings(self,filename):
        embeddings = dict()
        with open(filename) as f:
            for line in f:
                line = line.split()
                if len(line)==65:            
                    word = line[0]
                    coeffs = line[1:]
                    embeddings[word]= coeffs
        return embeddings

    def mat_builder(self,embed,tokenizer,ger_voc):#ger_tokenizer
        embedding_mat = np.zeros((ger_voc, 64))
        count=0
        for word,i in tokenizer.word_index.items():
            embedding_vec = embed.get(word)
            if embedding_vec is not None:
                embedding_mat[i]= embedding_vec
            else:
                count+=1
        return embedding_mat,count
                
    
    # bidirectional Encoder without attention
    def define_model(self,src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units,embedding_mat):
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
    
    def execute(self,no_units,batch_size,epochs,steps_per_epoch,val_steps):
#        filename = self.dataset_path
        doc = self.load_doc()
        #splitting by languages
        #lines = new_lines()
        pairs = self.to_pairs(doc)
        #print("pairs %d"%len(pairs))
        # cleaning
        cleaned_pairs = self.norm_pairs(pairs)
        
        raw_dataset = cleaned_pairs
        np.random.shuffle(raw_dataset)
        dataset = raw_dataset[:,:]
        train, test = dataset[:135000],dataset[135000:]
        
        eng_tokenizer = self.create_tokenizer(dataset[:, 0])
        eng_vocab_size = len(eng_tokenizer.word_index) + 1
        eng_length = self.max_length(dataset[:, 0])

        ger_tokenizer = self.create_tokenizer(dataset[:, 1])
        ger_vocab_size = len(ger_tokenizer.word_index) + 1
        ger_length = self.max_length(dataset[:, 1])
        
        self.save_clean_data(eng_tokenizer, "./output/eng_tok.pkl")    
        self.save_clean_data(ger_tokenizer, "./output/ger_tok.pkl")    
    
        dict_vars = {"ger_maxlen": ger_length, "model_path":"./output/model_generator.h5","eng_maxlen":eng_length}
        self.save_clean_data(dict_vars, "./output/dict_vars.pkl")

        trainX = self.encode_sequences(ger_tokenizer, ger_length, train[:, 1])
        trainY = self.encode_sequences(eng_tokenizer, eng_length, train[:, 0])

        # prepare validation data
        testX = self.encode_sequences(ger_tokenizer, ger_length, test[:, 1])
        testY = self.encode_sequences(eng_tokenizer, eng_length, test[:, 0])
        
        embed = self.read_all_embeddings(self.word_vec_path)

        embedding_mat,counter = self.mat_builder(embed,ger_tokenizer,ger_vocab_size)
        
        model = self.define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, no_units) # vocab size is total number of words, length is max sentence length
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        filegen = self.model_path
        checkpt = ModelCheckpoint(filegen, monitor = "val_acc", verbose = 2, save_best_only = True, mode="max")
        history=model.fit_generator(generator = self.new_gen(trainX,trainY, batch_size, eng_vocab_size),validation_data=self.new_gen(testX,testY,batch_size,eng_vocab_size), epochs=epochs,steps_per_epoch=steps_per_epoch, shuffle=False, validation_steps=val_steps , verbose=1, callbacks=[checkpt])




        