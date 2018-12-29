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

###############################################################################
class my_AdaBoost:
    def __init__(self, X, y, number_of_classifiers = 10, n_samples = 1024):
        self.X = X
        self.y = y

        self.number_of_classifiers = number_of_classifiers
        self.n_samples = n_samples
        #self.N = self.y.shape[0]
        #self.weights = [1 / self.N for _ in range(self.N)]
        self.count = 0

    def train_iteration(self):
        classifier_number = len(self.number_of_classifiers)

        best_precision = None
        best_error = None
        best_index = None
        for i in range(0,classifier_number):
            #get precision & assign it to 'best_precision' if it's better
            precision, error = self._evaluate_classifier(self.classifiers[j])
            if best_precision < precision:
                best_precision = precision
                best_error = error
                best_index

        #inversion
        if 0.5 < best_error: # this means that the best classifier is weak ( <0.5 means weak !)
            invert = -1
            best_error = 1 - best_error
        else:
            invert = 1

        # new classifier copy of the best classifer
        classifier = copy.copy(self.classifiers[best_index])
        alpha = 0.5 * invert * np.log((1 - best_error) / best_error)

        #Σ(a(t)*h(t)x)
        self.alphas.append(alpha)
        self.selected_classifiers.append(classifier)

        #get errors
        errors = (classifier.classify_data(self.data) != self.actual)

        # Δεν το πολυκαταλαβα αυτό
        if 0 > invert:
            errors = ~errors

        #ε = Σ (w(i) * error(i) )
        epsilon = np.sum(self.weights * errors)
        num_actual_0 = np.sum(self.actual[errors] == -1)
        num_actual_1 = np.sum(self.actual[errors] == 1)

        #W(t+1) = (W(t)/z) e^(-ah(t)y)
        self.weights[mistakes] *= 0.5 / sum_of_weights
        self.weights[~mistakes] *= 0.5 / (1 - sum_of_weights)

        '''
        #agreements = [-1 if e else 1 for e in errors]

        #a =1/2ln((1-ε)/ε) # ε= e του τ
        alpha = 0.5 * np.log((1 - epsilon)/ epsilon)

        #Σ(a(t)*h(t)x)
        self.alphas.append(alpha)
        self.selected_classifiers.append(classifier)

        #ε = Σ (w(i) * error(i) )
        epsilon = sum(self.weights * errors)
        num_actual_0 = np.sum(self.actual[errors] == -1)
        num_actual_1 = np.sum(self.actual[errors] == 1)

        #z = 2sqrt(ε(1- ε))
        z = 2 * np.sqrt(epsilon * ( 1 - epsilon))
        #W(t+1) = (W(t)/z) e^(-ah(t)y)
        self.weights = np.array([(weight / z )* np.exp(-1 * alpha * agreement)
                for weight, agreement in zip(self.weights, agreements)])
        '''

        
