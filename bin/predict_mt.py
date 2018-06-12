from common import encode_sequences
from __ import word_from_id ,predictions,fin_pred_model,punc_remover_lower
import argparse
from keras.models import load_model
import numpy as np
#import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("-s", metavar="string_to_translate",dest="src_sent", help="This is the string that must be translated to the other language", type=str)

args = parser.parse_args()

eng_tokenizer = np.load(open("./home/alindsharma/work/armaan/mactrans/output/eng_tok.pkl","rb"))
ger_tokenizer = np.load(open("./home/alindsharma/work/armaan/mactrans/output/ger_tok.pkl","rb"))
dict_vars = np.load(open("./home/alindsharma/work/armaan/mactrans/output/dict_vars.pkl",'rb'))

model = load_model(dict_vars["model_path"])        

ger_sent = args.src_sent
ger_sent = punc_remover_lower(ger_sent)
ger_sent = encode_sequences(ger_tokenizer, dict_vars['ger_maxlen'], ger_sent)
fin_source =list()
fin_source.append(ger_sent)
fin_pred_model(fin_source, eng_tokenizer, model)
