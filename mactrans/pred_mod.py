from keras.models import load_model
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#import tensorflow as tf
import string
#from keras.preprocessing.text import Tokenizer

class predicter(object):
    
    def __init__(self,src_sent,eng_tok_path,ger_tok_path,dict_vars_path):
        self.src_sent=src_sent
        self.eng_tok_path=eng_tok_path
        self.ger_tok_path=ger_tok_path                                        
        self.dict_vars_path = dict_vars_path
        
    #with tf.device("/gpu:0"):
    def encode_sequences(self,tokenizer, length, lines):
    	# integer encode sequences
        X = tokenizer.texts_to_sequences(lines)
        # pad sequences with 0 values
        X = pad_sequences(X, maxlen=length, padding='post')
        return X
        
    def word_from_id(self,integer, tokenizer):
        for word, i in tokenizer.word_index.items():
            if integer==i:
                return word
        return None
    
    def predictions(self,tokenizer, source, model):
        prediction = model.predict(source,verbose=0)[0]
        indices = [argmax(vector) for vector in prediction]
        output =[]    
        for i in indices:
            word = self.word_from_id(i,tokenizer)
            if word is None:
                break
            output.append(word)
        return " ".join(output) 
    
    def fin_pred_model(self,sources,tokenizer,model):
        for i,source in enumerate(sources):
    #        source = np.array(source)
            source = source.reshape((1,source.shape[0]))
            translation = self.predictions(tokenizer, source, model)
        print (translation)
    
    def punc_remover_lower(line):
        cleaned=[]    
        punc_rem = str.maketrans("","",string.punctuation)
        line =line.split()
        line = [word.translate(punc_rem) for word in line]   
        line = [word.lower() for word in line]
        cleaned.append(' '.join(line))
        return cleaned

        
    def execute(self):
        eng_tokenizer = np.load(open(self.eng_tok_path,"rb"))
        ger_tokenizer = np.load(open(self.ger_tok_path,"rb"))
        dict_vars = np.load(open(self.dict_vars_path,'rb'))
        
        model = load_model(dict_vars["model_path"])        
        
        ger_sent = self.src_sent
        ger_sent = self.punc_remover_lower(ger_sent)
        ger_sent = self.encode_sequences(ger_tokenizer, dict_vars['ger_maxlen'], ger_sent)
        self.fin_pred_model(ger_sent, eng_tokenizer, model)

