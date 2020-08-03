#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:48:05 2020

@author: inajim
"""
import numpy as np
import math

def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return max(0,x)

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    return 1*(x>0)

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1')
        self.hidden_to_output_weights = np.matrix('1 1 1')
        self.biases = np.matrix('0; 0; 0')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((-6, -3), -9), ((5, -8), -3), ((3, 2), 5), ((9, -5), 4), ((6, -1), 5), ((-5, 6), 1), ((1, -5), -4), ((6, 6), 12), ((2, 9), 11), ((-5, 0), -5)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]
        
        print("Starting params")
        print("Input to Hidden Layer weights" + str(self.input_to_hidden_weights))
        print("Hidden to Output Layer weights" + str(self.hidden_to_output_weights))
        print("Biases" + str(self.biases))

    def train(self, x1, x2, y):

        
        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        #Create matrix functions
        rectified_linear_matrix = np.vectorize(rectified_linear_unit)
        rectified_linear_matrix_derivative = np.vectorize(rectified_linear_unit_derivative)
        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.matmul(self.input_to_hidden_weights,input_values)+self.biases
        hidden_layer_activation = rectified_linear_matrix(hidden_layer_weighted_input)

        output =  np.matmul(self.hidden_to_output_weights, hidden_layer_activation)
        activated_output = output_layer_activation(output)

        ### Backpropagation ###

        # Compute gradients      
        output_layer_error = -(y-activated_output)*output_layer_activation_derivative(output)
        hidden_layer_error = np.multiply(output_layer_error,np.multiply(rectified_linear_matrix_derivative(hidden_layer_weighted_input),np.transpose(self.hidden_to_output_weights)))
        bias_gradients = hidden_layer_error
        hidden_to_output_weight_gradients = np.transpose(np.multiply(output_layer_error,hidden_layer_activation))
        input_to_hidden_weight_gradients = hidden_layer_error*np.transpose(input_values)

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases - self.learning_rate*bias_gradients
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate*input_to_hidden_weight_gradients
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate*hidden_to_output_weight_gradients

        
    def train_neural_network(self):

       for epoch in range(self.epochs_to_train):
           for x,y in self.training_points:
               self.train(x[0], x[1], y)
           print("\n Epoch" + str(epoch))
           print("Input to Hidden Layer weights" + str(self.input_to_hidden_weights))
           print("Hidden to Output Layer weights" + str(self.hidden_to_output_weights))
           print("Biases" + str(self.biases))

    # Run this to test your neural network implementation for correctness after it is trained
        

nn = NeuralNetwork()
nn.train_neural_network()