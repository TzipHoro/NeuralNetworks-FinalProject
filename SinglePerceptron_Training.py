# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:07:42 2022

@author: Ted
"""

# Library imports
from perceptron import Perceptron
import pandas as pd
import numpy as np
import time
np.set_printoptions(precision = 4, suppress = True)

# import data
data = pd.read_csv(r'data/train.csv') # data as a whole
training_data = data[0::2] # data selected for training
testing_data = data[1::2] # data selected for testing

# variable initialization
d = 0 # desired output
eta = 0.1 # gradient stepsize
input_vector = [0,0] # input values
weights = [0.8, 0.8] # initial weights
iterations = 1 # number of training sessions

# single perceptron initialization
P_0 = Perceptron(weights, 0)
y = [0] # variable to hold output

# defines the function to run the feedforward back propagation training method
def SP(P_0, input_vector, iterations, eta, y):
    # perceptron training session
    for i in range(0,iterations):
        # calculates activity value
        #P_0.calc_activity(input_vector)    # why is this called? this is called within method call below
        
        # calculates activation value
        P_0.calc_activation(input_vector)
        
        # sets the activation value
        y[0] = P_0.activation
        
        # calculates the delta value
        delta = (d-y[0])*y[0]*(1-y[0])
        
        # sets the delta value
        P_0.delta = delta
        
        # calculates the change in the weights
        P_0.set_delta_weights(input_vector, eta)
        
        # updates the weights
        P_0.update_weights()
        
        # update bias
        P_0.update_bias(eta)
        
        # prints the current iterate training session
        #print(y[0])

# FIRST TRAINING SESSION
# start time
t0 = time.time()

# Online Training
training_count = 30
for i in range(0,training_count):
    # training data row holding variable
    hold = training_data.iloc[i % 10]
    
    # variable update
    d = hold[3] # desired output
    input_vector = [hold[1],hold[2]] # input values
    
    # trains the perceptron based off inputed values
    SP(P_0, input_vector, iterations, eta, y)
    
# end time
t1 = time.time() - t0 # total time for training session
t_avg = t1 / float(training_count)

training_weights_1 = weights

# testing cycle    
# d = data.TACA[0] # desired output
# input_vector = [data.LAC[0],data.SOW[0]] # input values
    
# SP(P_0, input_vector, iterations, eta, y)

# # prints total error for the training session
# E = ((0.5)*(d - y[0])**2)

# activation value calculation
activation_1 = [] # activation values for tested values
total_error = 0
for i in range(0,10):
    # testing data row holding variable
    hold = testing_data.iloc[i]
    
    # variable update
    d = hold[3] # desired output
    input_vector = [hold[1],hold[2]] # input values
    
    # loads training weights
    P_0.weights = [training_weights_1[0],training_weights_1[1]]
    
    # tests the perceptron based off inputed values
    SP(P_0, input_vector, iterations, eta, y)
    
    # appends the activation value to a list
    activation_1.append(y[0])

    total_error += ((0.5)*(d - y[0])**2)

print("The time elapsed for training is", t1)
print(f"The average training time per instance: {t_avg}")
print("The final weights for this training are", training_weights_1)
print("Total error for this training was", total_error)
print("----")

# SECOND TRAINING SESSION
# variable initialization
d = 0 # desired output
input_vector = [0,0] # input values

# random weight generation
weights = np.random.uniform(size=2)

# single perceptron initialization
P_0 = Perceptron(weights, 0)

# start time
t0 = time.time()

# Online Training
training_count = 30
for i in range(0,training_count):
    # training data row holding variable
    hold = training_data.iloc[i % 10]
    
    # variable update
    d = hold[3] # desired output
    input_vector = [hold[1],hold[2]] #input values
    
    # trains the perceptron based off inputed values
    SP(P_0, input_vector, iterations, eta, y)
    
# end time
t1 = time.time() - t0 # total time for training session
t_avg = t1 / float(training_count)

training_weights_2 = weights

# testing cycle    
# d = data.TACA[0] # desired output
# input_vector = [data.LAC[0],data.SOW[0]] # input values
    
# SP(P_0, input_vector, iterations, eta, y)

# # prints total error for the training session
# E = ((0.5)*(d - y[0])**2)

# activation value calculation
total_error = 0
activation_2 = [] # activation values for tested values
for i in range(0,10):
    # testing data row holding variable
    hold = testing_data.iloc[i]
    
    # variable update
    d = hold[3] # desired output
    input_vector = [hold[1],hold[2]] # input values
    
    # loads training weights
    P_0.weights = [training_weights_2[0],training_weights_2[1]]
    
    # tests the perceptron based off inputed values
    SP(P_0, input_vector, iterations, eta, y)
    
    # appends the activation value to a list
    activation_2.append(y[0])

    total_error += ((0.5)*(d - y[0])**2)


print("The total time elapsed for training is", t1)
print(f"The average training time per instance: {t_avg}")
print("The final weights for this training are", training_weights_2)
print("The total error for this training was", total_error)
print('-----')


# THIRD TRAINING SESSION
# variable initialization
d = 0 # desired output
input_vector = [0,0] # input values

# random weight generation
weights = [] # initial weights
for i in range(2):
    # generates a random variable 
    hold = np.random.uniform()
    
    # adds a random amount of each training weights
    combined_hold = (training_weights_1[i] * hold) + (training_weights_2[i] * (1 - hold))
    
    # appends it to the list of weights
    weights.append(combined_hold)

# single perceptron initialization
P_0 = Perceptron(weights, 0)

# start time
t0 = time.time()

# Online Training
training_count = 30
for i in range(0,training_count):
    # training data row holding variable 
    hold = training_data.iloc[i % 10]
    
    # variable update
    d = hold[3] # desired output
    input_vector = [hold[1],hold[2]] #input values
    
    # trains the perceptron based off inputed values
    SP(P_0, input_vector, iterations, eta, y)
    
# end time
t1 = time.time() - t0 # total time for training session
t_avg = t1 / float(training_count)

training_weights_3 = weights

# testing cycle    
# d = data.TACA[0] # desired output
# input_vector = [data.LAC[0],data.SOW[0]] # input values
    
# SP(P_0, input_vector, iterations, eta, y)

# # prints total error for the training session
# E = ((0.5)*(d - y[0])**2)

# activation value calculation
total_error = 0
activation_3 = [] # activation values for tested values
for i in range(0,10):
    # testing data row holding variable
    hold = testing_data.iloc[i]
    
    # variable update
    d = hold[3] # desired output
    input_vector = [hold[1],hold[2]] # input values
    
    # loads training weights
    P_0.weights = [training_weights_3[0],training_weights_3[1]]
    
    # tests the perceptron based off inputed values
    SP(P_0, input_vector, iterations, eta, y)
    
    # appends the activation value to a list
    activation_3.append(y[0])

    total_error += ((0.5)*(d - y[0])**2)

print("The time elapsed for training is", t1)
print(f"The average training time per instance: {t_avg}")
print("The final weights for this training are", training_weights_3)
print("The total error for this training was", total_error)


# TRAINING WEIGHTS AND ACTIVATION EXPORT
# zips the training and testing files together before DataFrame creation
training_weights_zipped = list(zip(training_weights_1, training_weights_2, training_weights_3))
activation_zipped = list(zip(activation_1, activation_2, activation_3))

# creates a dataframe to store training weights and testing activation values
training_weights = pd.DataFrame(training_weights_zipped, columns = ["Training Set 1","Training Set 2","Training Set 3"])
testing_activation = pd.DataFrame(activation_zipped, columns = ["Testing Set 1","Testing Set 2","Testing Set 3"])

# exports the training weights and testing activation values as a csv file
training_weights.to_csv("training_weights.csv")
testing_activation.to_csv("testing_activation.csv")