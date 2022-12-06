# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 19:51:12 2022

@author: Ted
"""

# Library imports
import PerceptronClass as per
import pandas as pd
import numpy as np
import time
import random
np.set_printoptions(precision = 4, suppress = True)

# import data
data = pd.read_csv(r'data\train.csv') # data as a whole
training_data = data[0::2] # data selected for training

# variable initialization
d = 0 # desired output
eta = 0.1 # gradient stepsize
input_vector = [0,0] # input values
weights = [0.3, 0.3, 0.3, 0.3, 0.8, 0.8] # initial weights
iterations = 1 # number of training sessions
bias_update = 1 # boolean value that specifies if bias added in activity calculation

# hidden layer perceptron initialization
H_0 = per.perceptron([weights[0],weights[1]], 0)
H_1 = per.perceptron([weights[2],weights[3]], 0)

H_list = [H_0, H_1] # hidden layer list
y = [0] * len(H_list) # preallocates space for stored variables
delta_j = [0] * len(H_list) # preallocates space for delta_j values

# output lay perceptron initialization
O_0 = per.perceptron([weights[4],weights[5]], 0)

O_list = [O_0] # output layer list
z = [0] * len(O_list) # preallocates space for stored variables
delta_k = [0] * len(O_list) # preallocates space for delta_k values

# defines the function to run the feedforward back propagation training method
def FFBP(H_list, O_list, input_vector, iterations, eta, bias_update):
    # feedforward back propagation perceptron training session
    for i in range(0,iterations):
        # FEEDFORWARD
        for j in range(0,(len(H_list))):
            # calculates activity value
            H_list[j].calc_activity(input_vector, bias_update)
        
            # calculates activation value
            H_list[j].calc_activation(input_vector, bias_update)
            
            # sets the activation value
            y[j] = H_list[j].get_activation()
            
        for k in range(0,(len(O_list))):
            # calculates activity value
            O_list[k].calc_activity(y, bias_update)
        
            # calculates activation value
            O_list[k].calc_activation(y, bias_update)
            
            # sets the activation value
            z[k] = O_list[k].get_activation()
            
            # prints the current iterate training session
            print(z)
            
            # BACK PROPAGATION
            # calculates the delta value for the output layer
            delta_k[k] = (d - z[k]) * z[k] * (1 - z[k])
            
            # sets the delta value 
            O_list[k].set_delta(delta_k[k])
            
        for l in range(0,(len(H_list))):
        # calculates the delta value for the hidden layer before updating output weights
            delta_j[l] = (1 - y[l]) * y[l] * (sum(delta_k) * O_0.weights[l])
            
        for m in range(0,(len(O_list))):
            # calculates the change in the weights
            O_list[k].set_delta_weights(y, eta)
            
            # updates the weights
            O_list[k].update_weights()
            
            # update bias
            O_list[k].update_bias(eta)
            
        for n in range(0,(len(H_list))):
            # sets the delta value 
            H_list[n].set_delta(delta_j[n])
            
            # calculates the change in the weights
            H_list[n].set_delta_weights(input_vector, eta)
            
            # updates the weights
            H_list[n].update_weights()
            
            # update bias
            H_list[n].update_bias(eta)

# FIRST TRAINING SESSION
# start time
t0 = time.time()

# Online Training
for i in range(0,10):
    # training data row holding variable 
    hold = training_data.iloc[i]
    
    # variable update
    d = hold[3] # desired output
    input_vector = [hold[1],hold[2]] #input values
    
    # trains the perceptron based off inputed values
    FFBP(H_list, O_list, input_vector, iterations, eta, bias_update)
    
# end time
t1 = time.time() - t0 # total time for training session

training_weights_1 = H_list[0].weights + H_list[1].weights + O_list[0].weights

# testing cycle    
d = data.TACA[0] # desired output
input_vector = [data.LAC[0],data.SOW[0]] # input values
    
FFBP(H_list, O_list, input_vector, iterations, eta, bias_update)

# prints total error for the training session
E = ((0.5)*(d - z[0])**2)

print("The time elapsed for training is", t1)
print("The final weights for this training are", training_weights_1)
print("The big E error for this training was", E)

# SECOND TRAINING SESSION
# variable initialization
d = 0 # desired output
input_vector = [0,0] # input values

# random weight generation
weights = [] # initial weights
for i in range(6):
    # generates a random variable 
    hold = random.random()
    # appends it to the list of weights
    weights.append(hold)
    
iterations = 1 # number of training sessions
bias_update = 1 # boolean value that specifies if bias added in activity calculation

# hidden layer perceptron initialization
H_0 = per.perceptron([weights[0],weights[1]], 0)
H_1 = per.perceptron([weights[2],weights[3]], 0)

H_list = [H_0, H_1] # hidden layer list
y = [0] * len(H_list) # preallocates space for stored variables
delta_j = [0] * len(H_list) # preallocates space for delta_j values

# output lay perceptron initialization
O_0 = per.perceptron([weights[4],weights[5]], 0)

O_list = [O_0] # output layer list
z = [0] * len(O_list) # preallocates space for stored variables
delta_k = [0] * len(O_list) # preallocates space for delta_k values

# start time
t0 = time.time()

# Online Training
for i in range(0,10):
    # training data row holding variable 
    hold = training_data.iloc[i]
    
    # variable update
    d = hold[3] # desired output
    input_vector = [hold[1],hold[2]] #input values
    
    # trains the perceptron based off inputed values
    FFBP(H_list, O_list, input_vector, iterations, eta, bias_update)
    
# end time
t1 = time.time() - t0 # total time for training session

training_weights_2 = H_list[0].weights + H_list[1].weights + O_list[0].weights

# testing cycle    
d = data.TACA[0] # desired output
input_vector = [data.LAC[0],data.SOW[0]] # input values
    
FFBP(H_list, O_list, input_vector, iterations, eta, bias_update)

# prints total error for the training session
E = ((0.5)*(d - z[0])**2)

print("The time elapsed for training is", t1)
print("The final weights for this training are", training_weights_1)
print("The big E error for this training was", E)

# THIRD TRAINING SESSION
# variable initialization
d = 0 # desired output
input_vector = [0,0] # input values

# random weight generation
weights = [] # initial weights
for i in range(6):
    # generates a random variable 
    hold = random.random()
    
    # adds a random amount of each training weights
    combined_hold = (training_weights_1[i] * hold) + (training_weights_2[i] * (1 - hold))
    
    # appends it to the list of weights
    weights.append(combined_hold)
    
iterations = 1 # number of training sessions
bias_update = 1 # boolean value that specifies if bias added in activity calculation

# hidden layer perceptron initialization
H_0 = per.perceptron([weights[0],weights[1]], 0)
H_1 = per.perceptron([weights[2],weights[3]], 0)

H_list = [H_0, H_1] # hidden layer list
y = [0] * len(H_list) # preallocates space for stored variables
delta_j = [0] * len(H_list) # preallocates space for delta_j values

# output lay perceptron initialization
O_0 = per.perceptron([weights[4],weights[5]], 0)

O_list = [O_0] # output layer list
z = [0] * len(O_list) # preallocates space for stored variables
delta_k = [0] * len(O_list) # preallocates space for delta_k values

# start time
t0 = time.time()

# Online Training
for i in range(0,10):
    # training data row holding variable 
    hold = training_data.iloc[i]
    
    # variable update
    d = hold[3] # desired output
    input_vector = [hold[1],hold[2]] #input values
    
    # trains the perceptron based off inputed values
    FFBP(H_list, O_list, input_vector, iterations, eta, bias_update)
    
# end time
t1 = time.time() - t0 # total time for training session

training_weights_3 = H_list[0].weights + H_list[1].weights + O_list[0].weights

# testing cycle    
d = data.TACA[0] # desired output
input_vector = [data.LAC[0],data.SOW[0]] # input values
    
FFBP(H_list, O_list, input_vector, iterations, eta, bias_update)

# prints total error for the training session
E = ((0.5)*(d - z[0])**2)

print("The time elapsed for training is", t1)
print("The final weights for this training are", training_weights_1)
print("The big E error for this training was", E)