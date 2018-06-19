from mactrans.pred_mod import predicter
import argparse
# import tensorflow as tf


def parser_creator():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", metavar="string_to_translate",
                        dest="src_sent",
                        help="String to be translated to other language",
                        type=str)
    parser.add_argument("-e", metavar="Path to english tokenizer",
                        dest="eng_tok_path",
                        help="Path to trained english tokenizer.", type=str)
    parser.add_argument("-g", metavar="Path to german tokenizer",
                        dest="ger_tok_path",
                        help="Path to the trained german tokenizer.",
                        type=str)
    parser.add_argument("-d", metavar="Path to dict_vars",
                        dest="dict_vars_path",
                        help="Path to dict containing magic numbers.",
                        type=str)
    return parser

if __name__ == "__main__":
    parser = parser_creator()
    args = parser.parse_args()
    predict = predicter(args.src_sent, args.eng_tok_path,
                        args.ger_tok_path, args.dict_vars_path)
    predict.execute()
