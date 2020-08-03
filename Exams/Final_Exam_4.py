#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 03:37:13 2020

@author: inajim
"""

import numpy as np

def ReLU(x):
    return max(x,0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

w = np.array([0.1,-1,1])
x = -2
t_1 = 1
t_2 = -1

z_1 = w[0]*x
alpha_1 = ReLU(z_1)
y_1 = sigmoid(w[1]*alpha_1)
y_2 = sigmoid(w[2]*alpha_1)
loss = 1/2*(y_1-t_1)**2+1/2*(y_2-t_2)**2

print(alpha_1)
print(y_1)
print(y_2)
print(loss)


w = np.array([0.1+100,-1,1])
x = -2
t_1 = 1
t_2 = -1

z_1 = w[0]*x
alpha_1 = ReLU(z_1)
y_1 = sigmoid(w[1]*alpha_1)
y_2 = sigmoid(w[2]*alpha_1)
loss = 1/2*(y_1-t_1)**2+1/2*(y_2-t_2)**2

print(alpha_1)
print(y_1)
print(y_2)
print(loss)

for XX in range(-10,10):
    w = np.array([XX/10,-1,1])
    x = -2
    t_1 = 1
    t_2 = -1

    z_1 = w[0]*x
    alpha_1 = ReLU(z_1)
    y_1 = sigmoid(w[1]*alpha_1)
    y_2 = sigmoid(w[2]*alpha_1)
    loss = 1/2*(y_1-t_1)**2+1/2*(y_2-t_2)**2

    print(alpha_1)
    print(y_1)
    print(y_2)
    print(loss)