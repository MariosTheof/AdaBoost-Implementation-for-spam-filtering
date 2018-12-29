#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 18:43:32 2018

@author: marios
"""
import numpy as np
from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier


class MyAdaBoost():
    
    def __init__(self,
                 n_estimators=50,
                 learning_rate=1,
                 random_state=None):
        
        self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

        
    def my_fit(self, X, y):
        """Build a boosted classifier from the training set (X, y).
        
           
        """
        #fit
        sample_weight = np.ones(X.shape[0])/X.shape[0] #instead of len()
        for tree in range(self.n_estimator):
            estimator, sample_weight, estimator_weight= \
            self._boost(X,y, sample_weight)
            self.estimators_.append(estimator)
            self.estimator_weight_[tree]=estimator_weight
    
        
    def _boost(self, X, y, sample_weight):
        """
        """
        estimator = clone(self.base_estimator)
        #maybe tempEstimator = estimator
        # tempEstimator.fit(...)
        estimator.fit(X, y)
        
        pred_y = estimator.predict(X)
        
        indicator = np.ones(X.shape[0])*[pred_y!=y][0]
        
        error = np.dot(sample_weight, indicator) / np.sum(sample_weight)
        
        alpha = 0.5 * np.log((1-error)/error)
        
        new_sample_weight = sample_weight* np.exp(alpha*indicator)

        return estimator, new_sample_weight, alpha
            
    def predict(self, X):
        """
        """
        predicts = []
        for estimator in self.estimators_:
            pred = estimator.predict(X)
            pred[pred==0] = -1 ##
            predicts.append(pred)
            
        predicts = np.array(predicts) ##
        
        pr = np.sign(np.dot(self.estimator_weight_, predicts))
        pr[pr==-1] = 0
        return pr

    def score(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''
        #accuracy = TP+TN / n
        predictions = self.predict(x)
        n= x.shape[0]
        tp = np.sum(predictions * y)
        tn = np.sum((1-predictions)* (1-y))
        acc = (tp+tn)/n
        return acc
    
