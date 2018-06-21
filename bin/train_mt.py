# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 22:50:48 2018

@author: Armaan Puri
"""
from mactrans.train_mod import trainer
import argparse


def parser_creator():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', metavar="Prefix for model, toks and variables",
                        dest="prefix",
                        help="Prefix to store model,Tokenizers and Vars.",
                        type=str)

    parser.add_argument("-t", metavar="train_test_set",
                        dest="dataset_path",
                        help="Path to the formatted corpus",
                        type=str)

    parser.add_argument("-w", metavar="Word vec path",
                        dest="word_vec_path",
                        help="Path to preprocessed word embeddings",
                        type=str)

    parser.add_argument("-n", metavar="no of epochs",
                        dest="epochs",
                        default=50,
                        help='Iterations whole dataset is passed to the model',
                        type=int)

    parser.add_argument("-b", metavar="batch_size",
                        dest="batch_size",
                        default=125,
                        help="No of rows output from generator in single call",
                        type=int)

    parser.add_argument("-s", metavar='steps_per_epoch',
                        dest="steps_per_epoch",
                        default=1080,
                        help='No_of_exampleRows(train)/batch_size',
                        type=int)

    parser.add_argument("-v", metavar='validation_steps',
                        dest="val_steps",
                        default=128,
                        help='No_of_exampleRows(test)/batch_size',
                        type=int)

    parser.add_argument("-N", metavar="LSTM units",
                        dest="n_units",
                        default=512,
                        help="Size of the network (no of LSTM units)",
                        type=int)
    return parser

if __name__ == "__main__":
    parser = parser_creator()
    args = parser.parse_args()
    training_model = trainer(args.prefix, args.word_vec_path,
                             args.dataset_path)
    training_model.execute(args.n_units, args.batch_size,
                           args.epochs, args.steps_per_epoch, args.val_steps)
