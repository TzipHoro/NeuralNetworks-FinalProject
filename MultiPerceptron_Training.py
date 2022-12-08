# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 19:51:12 2022

@author: Ted
"""

from  perceptron import Perceptron
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# variable initialization
d = 0.7  # desired output
eta = 1.0  # gradient stepsize
input_vector = [1, 2]  # input values
weights = [0.3, 0.3, 0.3, 0.3, 0.8, 0.8]  # initial weights
iterations = 1  # number of training sessions

# hidden layer perceptron initialization
H_0 = Perceptron(weights[:2], 0)
H_1 = Perceptron(weights[2:4], 0)

H_list = [H_0, H_1]  # hidden layer list
y = [0] * len(H_list)  # preallocates space for stored variables
delta_j = [0] * len(H_list)  # preallocates space for delta_j values

# output lay perceptron initialization
O_0 = Perceptron(weights[4:], 0)

O_list = [O_0]  # output layer list
z = [0] * len(O_list)  # preallocates space for stored variables
delta_k = [0] * len(O_list)  # preallocates space for delta_k values


def FFBP(H_list, O_list, input_vector, iterations, eta, bias_update,
    delta_j, delta_k, O_0, y, z, d):
    """Function to run the feedforward back propagation training method"""
    for i in range(0, iterations):
        # FEEDFORWARD
        for j in range(0, len(H_list)):
            H_list[j].calc_activity(input_vector, bias_update)
            H_list[j].calc_activation(input_vector, bias_update)
            y[j] = H_list[j].activation
            
        for k in range(0, len(O_list)):
            O_list[k].calc_activity(y, bias_update)
            O_list[k].calc_activation(y, bias_update)
            z[k] = O_list[k].activation
            print(z)

            # BACK PROPAGATION
            # calculates the delta value for the output layer
            delta_k[k] = (d - z[k]) * z[k] * (1 - z[k])

            # sets the delta value 
            O_list[k].delta = delta_k[k]
            
        for l in range(0, len(H_list)):
            # calculates the delta value for the hidden layer before updating output weights
            delta_j[l] = (1 - y[l]) * y[l] * (sum(delta_k) * O_0.weights[l])
            
        for m in range(0, len(O_list)):
            O_list[k].set_delta_weights(y, eta)
            O_list[k].update_weights()
            O_list[k].update_bias(eta)
            
        for n in range(0, len(H_list)):
            H_list[n].delta = delta_j[n]
            H_list[n].set_delta_weights(input_vector, eta)
            H_list[n].update_weights()
            H_list[n].update_bias(eta)


def main():

    # variable initialization
    d = 0.7   # desired output 
    eta = 1.0   # gradient stepsize
    input_vector = [1, 2]   # input values
    weights = [0.3, 0.3, 0.3, 0.3, 0.8, 0.8]   # initial weights
    iterations = 1   # number of training sessions
    bias_update = 1  # flag: bias added in activity calculation

    # hidden layer perceptron initialization
    H_0 = Perceptron([weights[0], weights[1]], 0)
    H_1 = Perceptron([weights[2], weights[3]], 0)

    H_list = [H_0, H_1]  # hidden layer list
    y = [0] * len(H_list)  # preallocates space for stored variables
    delta_j = [0] * len(H_list)  # preallocates space for delta_j values

    # output lay perceptron initialization
    O_0 = Perceptron([weights[4], weights[5]], 0)

    O_list = [O_0]  # output layer list
    z = [0] * len(O_list)   # preallocates space for stored variables
    delta_k = [0] * len(O_list)  # preallocates space for delta_k values

    # Online Training
    for i in range(1, 31):
        # even training cycle
        if (i % 2) == 0:
            # variable update
            d = 0.05 # desired output
            input_vector = [-1,-1] # input values
            iterations = 1 # number of training sessions
            bias_update = 1 # boolean value that specifies if bias is updated
        
        # odd training cycle    
        else:
            # variable update
            d = 0.9 # desired output
            input_vector = [1,1] # input values
            iterations = 1 # number of training sessions
            bias_update = 1 # boolean value that specifies if bias is updated
        
        # trains the perceptron based off inputed values
        FFBP(H_list, O_list, input_vector, iterations, eta, bias_update)
        
    # testing cycle    
    d = 0.9 # desired output
    input_vector = [1, 1] # input values
        
    FFBP(H_list, O_list, input_vector, iterations, eta, bias_update,
     delta_j, delta_k, O_0, y, z, d)

    # prints total error for the training session
    E = ((0.5)*(d - z[0])**2)


if __name__ == '__main__':
    main()

