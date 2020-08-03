#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:48:11 2020

@author: inajim
"""

import numpy as np
from sklearn.svm import SVC



#y = np.array([-1,-1,-1,-1,-1,1,1,1,1,1])
#x = np.array([[0,0],[2,0],[3,0],[0,2],[2,2],[5,1],[5,2],[2,4],[4,4],[5,5]])

#clf = SVC(C = 1000000000000, kernel = 'linear')
#clf.fit(x, y) 
#print(clf.coef_)
#print(clf.intercept_)

y = np.array([-1,-1,-1,-1,-1,1,1,1,1,1])
x = np.array([[0,0],[2,0],[1,1],[0,2],[3,3],[4,1],[5,2],[1,4],[4,4],[5,5]])

clf = SVC(C = 1000000000000, kernel = 'poly', degree = 2)
clf.fit(x, y) 
print(clf.dual_coef_)




def hinge_loss(x,y,theta,theta_0):
    return max(0,1-y*np.dot(x,theta)-theta_0)

def hinge_loss_derivative_theta(x,y,theta, theta_0):
    if 1-y*np.dot(x,theta)-theta_0>0:
        return -y*x
    else:
        return 0
    
def hinge_loss_derivative_theta0(x,y,theta, theta_0):
    if 1-y*np.dot(x,theta)-theta_0>0:
        return -1
    else:
        return 0

#hinge_loss_matrix = np.vectorize(hinge_loss)
#hinge_loss_derivative_matrix = np.vectorize (hinge_loss_derivative)
    

def SVM(x,y):
    n = x.shape[0]
    d = x.shape[1]
    theta = np.zeros(d)
    theta_0 = 0

    rounds = 0
    
    loss = 0
    derivative_theta = 0
    derivative_theta0 = 0
    for i in range(n):
        loss = loss + hinge_loss(x[i,:],y[i],theta,theta_0)/n
        derivative_theta = derivative_theta + hinge_loss_derivative_theta(x[i,:],y[i],theta,theta_0)/n
        derivative_theta0 = derivative_theta0 + hinge_loss_derivative_theta0(x[i,:],y[i],theta,theta_0)/n
    loss = loss + theta    
    while loss > 0:
        theta = theta - derivative_theta
        theta_0 = theta_0 - derivative_theta0
            
        loss = 0
        derivative_theta = 0
        derivative_theta0 = 0
        for i in range(n):
            loss = loss + hinge_loss(x[i,:],y[i],theta,theta_0)/n
            derivative_theta = derivative_theta + hinge_loss_derivative_theta(x[i,:],y[i],theta,theta_0)/n
            derivative_theta0 = derivative_theta0 + hinge_loss_derivative_theta0(x[i,:],y[i],theta,theta_0)/n
        print(loss)
        
    return theta,theta_0
    