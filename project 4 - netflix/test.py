import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

initial_mixture, post = common.init(X,K,seed)
print(X)
print(initial_mixture)
post,LL = em.estep(X, initial_mixture)
print(post)
print(LL)

new_mixture = em.mstep(X,post,initial_mixture,0.25)
print(new_mixture)
