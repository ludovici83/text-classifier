#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import nltk
import string 
from nltk.corpus import stopwords
import io
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
import os.path
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

lemmatizer = WordNetLemmatizer() 
def process_text(text):
    text = text.lower()
    # removing punctuation symbols
    text = text.translate(str.maketrans('', '', string.punctuation))
    # removing stopwords
    list_stopwords = stopwords.words('english')
    text_tokens = word_tokenize(text)
    tokens_without_sw = [lemmatizer.lemmatize(word) for word in text_tokens if not word in list_stopwords]
    return ' '.join(tokens_without_sw)

class_dict = {0:"exploration",1:"headhunters",2:"intelligence",3:"logistics",4:"politics",5:"transportation",6:"weapons"}


def classify(model,args):
    try:
        f = open(model,"rb")
    except FileNotFoundError:
        print("You need first to execute the train script to save the trained classification model")
    
    try:
        f = open("count_vectorizer","rb")
    except IOError:
        print("You need first to execute the train script to save the trained count_vectorizer already fitted to the corpus-vocabulary")
    
    clf = pickle.load(open(model, 'rb'))
    count_vect = pickle.load(open("count_vectorizer", 'rb'))
    
    file_list = []
    texts_list = []
    root = os.getcwd()
    for file in args:
        file_path = os.path.join(root,file)
        #file_text = open(file_path, "r").read() 
        file_text = io.open(file_path, mode="r", encoding="ISO-8859-1").read()
        texts_list.append(file_text)
        file_list.append(file)
        
    texts_list = [process_text(t) for t in texts_list]
    
    X_texts        = pd.Series(texts_list)
    X_texts_counts = count_vect.transform(X_texts)
    
    prediction = clf.predict(X_texts_counts)
    
    for i in range(len(file_list)):
        print(file_list[i],class_dict[int(prediction[i])])





