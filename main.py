#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 23:12:33 2018

@author: marios
"""

#Based on this paper : http://rob.schapire.net/papers/explaining-adaboost.pdf
#Using enron spam database

import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score,accuracy_score

from sklearnInspired import MyAdaBoost

from nltk.corpus import stopwords


###############################################################################
#read data in CSV format according to your PC's address

#Read spam folder
df_spam = pd.DataFrame(columns=['observation'])
spam_path = "/home/marios/Desktop/AdaBoostImplementation/enron1/spam"

for file in os.listdir(spam_path):
    #if os if os.path.isdir(directory):
    #for filename in os.listdir(directory):
    with open(os.path.join(spam_path, file), encoding="utf-8",errors='ignore') as f:
        observation = f.read()
        current_df = pd.DataFrame({'observation': [observation]})
        df_spam = df_spam.append(current_df, ignore_index=True)


#Read ham folder
df_ham = pd.DataFrame(columns=['observation'])
ham_path = "/home/marios/Desktop/AdaBoostImplementation/enron1/ham"

for file in os.listdir(ham_path):
    #if os if os.path.isdir(directory):
    #for filename in os.listdir(directory):
    with open(os.path.join(ham_path, file), encoding="utf-8",errors='ignore') as f:
        observation = f.read()
        current_df = pd.DataFrame({'observation': [observation]})
        df_ham = df_ham.append(current_df, ignore_index=True)
        
        
###############################################################################
#DataFraming 'preprocessing'
#Rename 'observation' columns to 'Text'
df_spam.rename(columns={'observation':'Text'},inplace=True)
df_ham.rename(columns={'observation':'Text'},inplace=True)

#Add Class column to each dataFrame
df_spam['Class'] = 0
df_ham['Class'] = 1

#Combine into one dataFrame called 'data'
data = df_spam.append(df_ham, ignore_index=True)


###############################################################################
"""
Use this to download the nltk stopwords: 
    import nltk
    nltk.download('stopwords')

Also, make sure that you have imported the required module, using : 
    from nltk.corpus import stopwords

"""

#Removing stopwords of English
stopset = set(stopwords.words("english"))

##############################################################################

#Initialising Count Vectorizer
vectorizer = CountVectorizer(stop_words=stopset,binary=True)
#vectorizer = CountVectorizer()

#fit and transform
X = vectorizer.fit_transform(data.Text)
# Extract target column 'Class'
y = data.Class


#Performing test train Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, train_size=0.70, random_state=None)


##############################################################################
#train classifier
#clf = AdaBoostClassifier(n_estimators=100)
'''
clf = DecisionTreeClassifier(max_depth = 1)
clf.fit(X_train,y_train)
clf.score(X_train,y_train)
'''
##############################################################################

my_clf = MyAdaBoost()
my_clf.my_fit(X_train, y_train)

my_clf.score(X_train, y_train)
##############################################################################

#make prediction
y_pred = my_clf.predict(X_test)
#prediction f1_score
f1_score(y_test, y_pred)
#prediciton accuracy score
accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

##############################################################################


