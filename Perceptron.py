# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:05:24 2022

@author: Ted
"""

# Library imports
import numpy as np


class Perceptron:
    # defining the perceptron class and fields associated with it
    def __init__(self, initial_weights: list, bias: float = None):
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

    def calc_activity(self, input_vector: list):
        # initializes input and activity values
        activity = 0

        # for loop to calculate activity based on input/weight pairs
        for i in range(0, self.vector_length):
            activity += input_vector[i] * self.weights[i]

        # check if the bias is being added
        if self.bias is None:
            # adds bias value to all activity terms
            activity += self.bias

        # updates activity value
        self.activity = activity

    def calc_activation(self, input_vector: list):
        # calculates the activity value
        self.calc_activity(input_vector)

        # calculates activation value based on activation function
        self.activation = 1.0 / (1.0 + np.exp(-1 * self.activity))

    def get_activation(self):
        # returns the current activation value
        return self.activation

    def set_delta(self, delta: float):
        # updates the current delta value
        self.delta = delta

    def get_delta(self):
        # returns the current delta value
        return self.delta

    def set_delta_weights(self, input_vector, eta):
        # calcuates the change to the weights
        for i in range(0, self.vector_length):
            self.delta_weights[i] = input_vector[i] * eta * self.delta

    def update_weights(self):
        # updates the weights for the perceptron
        for i in range(0, self.vector_length):
            self.weights[i] = self.weights[i] + self.delta_weights[i]

    def update_bias(self, eta):
        # updates the bias value
        if self.bias is not None:
            self.bias = self.bias + eta * self.delta * 1
