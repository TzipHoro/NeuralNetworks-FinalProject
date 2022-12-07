# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:05:24 2022

@author: Ted
"""

# TODO: (M.Kouremetis) I think we should use numpy matrix operations versus iterating over lists


import numpy as np


class Perceptron:
    def __init__(self, initial_weights: list, bias: float):
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
        
    def calc_activity(self, input_vector: list, bias: bool):
        """ """
        input = input_vector
        A = 0
        for i in range(0, self.vector_length):
            A += input[i] * self.weights[i]
        if bias == 1:    
            A += self.bias
        self.activity = A
        
    def calc_activation(self, input_vector: list, bias: bool):
        """ """
        self.calc_activity(input_vector, bias)
        self._activation = 1.0/(1.0 + np.exp(-1*self.activity))
    
    
    def set_delta_weights(self, input_vector: list, eta: float):
        """ """
        input = input_vector
        # calcuates the change to the weights
        for i in range(0, self.vector_length):
            self.delta_weights[i] = input[i] * eta * self._delta
        
    def update_weights(self):
        """ """
        for i in range(0, self.vector_length):
            self.weights[i] += self.delta_weights[i]
            
    def update_bias(self, eta: float):
        """ """
        self.bias += eta * self.delta * 1
