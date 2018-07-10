# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 03:19:20 2018

@author: Armaan Puri
"""
import argparse
from gensim.models import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument('-b', metavar="path to .bin",
                    dest="bin_w2v",
                    help="path to Word2Vec in binary format",
                    type=str)
parser.add_argument("-t", metavar="path to .txt",
                    dest="txt_w2v",
                    help="Path to Word2Vec in txt format",
                    type=str)
args = parser.parse_args()


def read_all(filename):
    model_w2v = Word2Vec.load(filename, mmap="r")
    return model_w2v

x = read_all(args.bin_w2v)
tab = open(args.txt_w2v, 'w')
for i in x.wv.vocab:
    mystr = str(i) + " " + (str(list(x[i])).strip("[")).strip("]")
    mystr = (mystr.strip("\n")).replace(",", "") + "\n"
#    print(mystr)
    tab.write(mystr)

tab.close()
