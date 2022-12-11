# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:05:24 2022

@author: Ted
"""

# TODO: (M.Kouremetis) I think we should use numpy matrix operations versus iterating over lists


import numpy as np


class Perceptron:
    def __init__(self, initial_weights: list, bias: float = None):
        self.weights = initial_weights 
        self.bias = bias 
        self._delta = 0 
        self.delta_weights = [0] * len(self.weights)
        self.activity = 0 
        self._activation = 0 
        # creates a vector to know the number of inputs associated with the weights
        self.vector_length = len(self.weights)

    @property
    def activation(self):
        return self._activation
    
    @activation.setter
    def activation(self, value):
        self._activation = value

    @property
    def delta(self):
        return self._delta
    
    @delta.setter
    def delta(self, value):
        self._delta = value
        
    def calc_activity(self, input_vector: list):
        #activity = np.dot(np.array(input_vector), np.array(self.weights))   # when tested, this was actually slower performance
        activity = 0
        for i in range(0, self.vector_length):
             activity += input_vector[i] * self.weights[i]
        if self.bias is not None:
            activity += self.bias
        self.activity = activity
        
    def calc_activation(self, input_vector: list):
        """ """
        self.calc_activity(input_vector)
        self._activation = 1.0/(1.0 + np.exp(-1*self.activity))
    
    def set_delta_weights(self, input_vector: list, eta: float):
        """ 

        PERFORMANCE NOTE: Could do vector addition here with numpy.
        1) given minimal weights, performance speed up not observed
        2) interestingly using numpy in this regard causes loss of
        precision, apparently known issue (https://stackoverflow.com/questions/21165745/precision-loss-numpy-mpmath)
        """
        input = input_vector
        # calculates the change to the weights
        for i in range(0, self.vector_length):
            self.delta_weights[i] = input[i] * eta * self._delta
        
    def update_weights(self):
        """

        PERFORMANCE NOTE: Could do vector addition here with numpy.
        1) given minimal weights, performance speed up not observed
        2) interestingly using numpy in this regard causes loss of
        precision, apparently known issue (https://stackoverflow.com/questions/21165745/precision-loss-numpy-mpmath)
        """
        for i in range(0, self.vector_length):
            self.weights[i] += self.delta_weights[i]
            
    def update_bias(self, eta: float):
        """ """
        if self.bias is not None:
            self.bias += eta * self.delta * 1
