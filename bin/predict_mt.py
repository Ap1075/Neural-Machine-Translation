from mactrans.pred_mod import predicter
import argparse
# import tensorflow as tf
import os


def parser_creator():

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", metavar="string_to_translate",
                        dest="src_sent",
                        help="String to be translated to other language",
                        type=str)

    parser.add_argument("-P", metavar="prefix for pickle",
                        dest="prefix",
                        help="Prefix to .h5 and pickle containing\
                              tokenizers, variables",
                        type=str)
    return parser

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    parser = parser_creator()
    args = parser.parse_args()
    predict = predicter(args.src_sent, args.prefix)
    predict.execute()
