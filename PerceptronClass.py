# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:05:24 2022

@author: Ted
"""

import numpy as np


class Perceptron:
    def __init__(self, initial_weights, bias):
        self.weights = initial_weights 
        self.bias = bias 
        self.delta = 0 
        self.delta_weights = [0] * len(self.weights)
        self.activity = 0 
        self.activation = 0 
        # creates a vector to know the number of inputs associated with the weights
        self.vector_length = len(self.weights)
        
    def calc_activity(self, input_vector, bias_bool):
        """ """
        input = input_vector
        A = 0
        # for loop to calculate activity based on input/weight pairs
        for i in range(0, self.vector_length):
            A += input[i] * self.weights[i]
        if bias_bool == 1:    
            A += self.bias
        self.activity = A
        
    def calc_activation(self, input_vector, bias_bool):
        """ """
        self.calc_activity(input_vector, bias_bool)
        self.activation = 1.0/(1.0 + np.exp(-1*self.activity))
    
    def get_activation(self):
        """ """
        return(self.activation)
    
    def set_delta(self, delta):
        """ """
        self.delta = delta
        
    def get_delta(self):
        """ """
        return(self.delta)
    
    def set_delta_weights(self, input_vector, eta):
        """ """
        input = input_vector
        # calcuates the change to the weights
        for i in range(0, self.vector_length):
            self.delta_weights[i] = input[i] * eta * self.delta
        
    def update_weights(self):
        """ """
        for i in range(0, self.vector_length):
            self.weights[i] = self.weights[i] + self.delta_weights[i]
            
    def update_bias(self, eta):
        """ """
        self.bias = self.bias + eta * self.delta * 1