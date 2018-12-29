#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 23:12:33 2018

@author: marios
"""

#Based on this paper : http://rob.schapire.net/papers/explaining-adaboost.pdf
#Useful! : https://www.youtube.com/watch?v=UHBmv7qCey4&list=PLUl4u3cNGP63gFHB6xb-kVBiQHYe_4hSi&index=18
#Using enron spam database

import numpy as np

def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)
           
           
           
           
           
           

def adaboost_est(y_train, X_train, y_test, X_test, M, clf):
    n_train, n_test = X_train.shape[0], X_test.shape[0] #instead of len()
    # Initialize weights
    weights = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
        for i in range(M):
        clf.fit(X_train, y_train)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        #
        errors = [int(x) for x in (pred_train_i != y_train)]
        #
        agreement = [x if x == 1 else -1 for x in errors]
        #
        epsilon = np.dot(weights,errors) / sum(weights) # έχω και άλλη για το 'ε'
        #a =1/2ln((1-ε)/ε) # ε= e του τ
        alpha = 0.5 * np.log((1 - epsilon)/ epsilon) 
        #new weights        
        weights = np.multiply(weights, np.exp([float(x) * alpha for x in agreement]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [x * alpha for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [x * alpha for x in pred_test_i])]
        
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
     
    # Return error rate in train and test set
    return get_error_rate(pred_train, y_train), \
           get_error_rate(pred_test, y_test)
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           