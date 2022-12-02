# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:05:24 2022

@author: Ted
"""

# Library imports
import numpy as np

class perceptron(object):
    # defining the perceptron class and fields associated with it
    def __init__(self, initial_weights, bias):
        # weight vector initialization with initial weight vector
        self.weights = initial_weights 
        
        # initial bias value
        self.bias = bias 
        
        # initial delta value
        self.delta = 0 
        
        # initial weight change
        self.delta_weights = [0] * len(self.weights)
        
        # initial activity value
        self.activity = 0 
        
        # initial activation value
        self.activation = 0 
        
        # creates a vector to know the number of inputs associated with the weights
        self.vector_length = len(self.weights)
        
    def calc_activity(self, input_vector, bias_bool):
        # initializes input and activity values
        input = input_vector
        A = 0
        
        # for loop to calculate activity based on input/weight pairs
        for i in range(0, self.vector_length):
            A += input[i] * self.weights[i]
        
        # boolean value to check if the bias is being added
        if bias_bool == 1:    
            # adds bias value to all activity terms
            A += self.bias
        
        # updates activity value
        self.activity = A
        
    def calc_activation(self, input_vector, bias_bool):
        # calculates the activity value
        self.calc_activity(input_vector, bias_bool)
        
        # calculates activation value based on activation function
        self.activation = 1.0/(1.0 + np.exp(-1*self.activity))
    
    def get_activation(self):
        # returns the current activation value
        return(self.activation)
    
    def set_delta(self, delta):
        # updates the current delta value
        self.delta = delta
        
    def get_delta(self):
        # returns the current delta value
        return(self.delta)
    
    def set_delta_weights(self, input_vector, eta):
        # initializes input value
        input = input_vector
        
        # calcuates the change to the weights
        for i in range(0, self.vector_length):
            self.delta_weights[i] = input[i] * eta * self.delta
        
    def update_weights(self):
        # updates the weights for the perceptron
        for i in range(0, self.vector_length):
            self.weights[i] = self.weights[i] + self.delta_weights[i]
            
    def update_bias(self, eta):
        # updates the bias value
        self.bias = self.bias + eta * self.delta * 1