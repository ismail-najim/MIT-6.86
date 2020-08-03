#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 03:16:02 2020

@author: inajim
"""

from sklearn.svm import SVC
import numpy as np

pi = np.pi
X = np.array([pi, 2*pi, 3*pi, 4*pi, 5*pi, 6*pi, 7*pi, 8*pi, 9*pi, 10*pi ])
X = X.reshape(-1,1)
Y = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])

def cos_kernel(X,Y):
    return np.cos(X)*np.cos(Y)

def sin_kernel(X,Y):
    return np.sin(X)*np.sin(Y)

#print(np.cos(X))
#print(np.sin(X))

square =  SVC(kernel = 'poly', degree = 2)
twenty =  SVC(kernel = 'poly', degree = 20)
rbf =  SVC(kernel = 'rbf', gamma = 10)
#cos =  SVC(kernel = cos_kernel)
#sin =  SVC(kernel = sin_kernel)

square.fit(X,Y)
twenty.fit(X,Y)
#cos.fit(X,Y)
#sin.fit(X,Y)
rbf.fit(X,Y)

print(square.predict(X))
print(twenty.predict(X))
print(rbf.predict(X))