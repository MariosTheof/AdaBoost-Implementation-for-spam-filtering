#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:35:23 2019

@author: marios
"""

import math

class DecisionTree:
    def __init__(self,X ,y , verbose=False):
        self.X = X
        self.y = y
        self.verbose = verbose
        
    def build_tree(self,data, attributes, target, recursion):
        if data.shape[0] == 0:
            return None
        
        recursion += 1
        data = data[:] # depending on data. it will create a new view to the same data if NumPy array
        vals = [record[attributes.index(target)] for record in data]
        default = self.majority(data, attributes, target)
            
    
    
    def majority(self,data,attributes,target):
        '''
        return attribute whose occurence is max ??
        '''
        value_frequency = {}
        i = attributes.index(target)
        
        for val in data:
            if val[index] in value_frequency:
                value_frequency[val[i]] += 1
            else:
                value_frequency[val[i]] = 1
                
        max_attribute = 0
        major = ""
        for key in value_frequency.keys():
            if value_frequency[key] == max_attribute:
                major == 'Tie'
                
            if value_frequency[key] > max_attribute:
                major = key
                max_attribute = value_frequency[key]
                
        return major
    
    
    def entropy(self, attributes, data, targetAttr):
        '''
        Calculate entropy
        '''
        value_frequency = {}
        data_entropy = 0.0
        
        #find index of attribute
        i=0
        i = attributes.index(targetAttr)
    
        # Calculate the frequency of each of the values in the target attr
        for entry in data:
            if entry[i] in value_frequency:
                value_frequency[entry[i]] += 1.0
            else:
                value_frequency[entry[i]]  = 1.0

        # Calculate the entropy of the data for the target attr
        for freq in value_frequency.values():
            data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
        return data_entropy
    
    def gain(self, attributes, data, attr, targetAttr):
        '''
        Calculate gain
        '''
        value_frequency = {}
        subset_entropy = 0.0
        
        #find index of attribute
        i = attributes.index(attr)
        
        #calculate frequency of each attribute
        for entry in data:
            if entry[i] in value_frequency:
                value_frequency[entry[i]] += 1.0
            else:
                value_frequency[entry[i]] = 1.0
                
        #calculate the sum of the entropy    
        for val in value_frequency.keys():
            value_propability = value_frequency[val] / sum(value_frequency.values())
            data_subset = [entry for entry in data if entry[i] == val]
            subset_entropy += value_propability * self.entropy(attributes, data_subset, targetAttr)

        return (self.entropy(attributes, data,targetAttr) - subset_entropy)

    
    def find_best_attribute(self,data, attributes, target):
        '''
        Return attribute with max gain
        '''
        best_attr = None
        max_info_gain = 0
        for attr in attributes:
            
            new_info_gain = self.gain(attributes, data, attr, target)
            if new_info_gain > max_info_gain:
                max_info_gain = new_info_gain
                best_attr = attr
            
        return best_attr
        





                            