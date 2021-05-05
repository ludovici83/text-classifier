import train as t
import classify as c

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-a','--action', required=True)
parser.add_argument('-f','--folder_path', required=False)

parser.add_argument('-m','--model_path',required=False)
parser.add_argument('-t','--text_path',required=False,nargs="*")

args = vars(parser.parse_args())


if args["action"]=="train":
    if args["folder_path"] is None:
        print("must provide a file name to save the output trained model")
    t.train( args["folder_path"] ) 

if args["action"]=="classify":
    if args["model_path"] is None:
        print("must provide path of trained model, first execute train script")
    if args["text_path"] is None:
        print("must provide at least one path to a text to classify")
    c.classify(args["model_path"],args["text_path"])