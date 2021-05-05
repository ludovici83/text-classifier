#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import nltk
import string 
import io
from nltk.corpus import stopwords
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


def train(folder_path):
    path = os.path.join(os.getcwd(),folder_path)
    #categories = os.listdir( os.getcwd(),folder_path )
    categories = os.listdir(path)
    if ".DS_Store" in categories:
        categories.remove(".DS_Store")

    list_texts = []
    for categ in categories:
        category_folder = os.path.join(path, categ)
        category_files = os.listdir(category_folder)
        for f in category_files:
            file_path = os.path.join(category_folder,f)
            #file_text = open(file_path, "r").read() 
            file_text = io.open(file_path, mode="r", encoding="ISO-8859-1").read()
            row = (file_text , categ) 
            list_texts.append(row)
    
    
    df_texts = pd.DataFrame(list_texts,columns=["text","class_label"])
    del list_texts
    df_texts = df_texts.drop_duplicates()
    
    
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
    
    df_texts["text_processed"] = df_texts.text.apply(process_text)
    print("text has been successfully processed")
    count_vect       = CountVectorizer()
    
    def class_to_num(category):
        if category=="exploration":
            return 0
        if category=="headhunters":
            return 1
        if category == "intelligence":
            return 2
        if category == "logistics":
            return 3
        if category == "politics":
            return 4
        if category == "transportation":
            return 5
        if category=="weapons":
            return 6
        
    df_texts["target"] = df_texts["class_label"].apply(class_to_num)
    
    X = df_texts["text_processed"]
    y = df_texts["target"]
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    
    X_train_counts = count_vect.fit_transform(X_train)
    X_test_counts  = count_vect.transform(X_test)
    
    print("fitting the model")
    clf_naive_bayes   = MultinomialNB().fit(X_train_counts, y_train)
    predicted_naive_bayes   = clf_naive_bayes.predict(X_test_counts)
    accuracy = np.mean(predicted_naive_bayes == y_test)
    
    print("the accuracy of the model on the test data is: "+str(accuracy))
    filename = "model"
    pickle.dump(clf_naive_bayes, open(filename, 'wb'))
    print("trained model successfully saved")
    pickle.dump(count_vect, open("count_vectorizer", 'wb'))
    
