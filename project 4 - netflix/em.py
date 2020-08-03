"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    log_likl = np.zeros((n, K))
    log_post = np.zeros((n, K))
    post = np.zeros((n, K))
    
    #Calculate Hu and Cu
    Hu_m = np.isin(X,0)*1
    Cu_m = np.ones((n,d))-Hu_m
    Cu_v = np.sum(Cu_m,axis = 1)
    
    # Calculate likelihood for each point
    for i in range(n):
        for j in range(K):
            log_likl[i,j]= np.log(mixture.p[j] + 1e-16) + np.log((1/(2*np.pi*mixture.var[j])**(Cu_v[i]/2))*np.exp(-1/2*(1/mixture.var[j])*np.linalg.norm(np.multiply(X[i,:],Cu_m[i,:])-np.multiply(mixture.mu[j,:],Cu_m[i,:]))**2))
    
    for i in range(n):
        for j in range(K):
            log_post[i,j] = log_likl[i,j]-logsumexp(log_likl[i,:])
    
    post = np.exp(log_post)
    
    likl = np.exp(log_likl)
    LL = 0
    for i in range(n):
        LL = LL + np.log(np.sum(likl[i,:]))
    return post, LL




def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    # Define parameters and dimensions
    n, d = X.shape
    mu = mixture.mu.copy()
    _post = np.copy(np.transpose(post))
    K, _ = _post.shape
    p = np.zeros(K)
    var = np.zeros(K)
   
    
    
    #Calculate Hu and Cu
    Hu_m = np.isin(X,0)*1
    Cu_m = np.ones((n,d))-Hu_m
    Cu_v = np.sum(Cu_m,axis = 1)

    print(_post.shape)
    
    # Compute p
    p = post.sum(axis=0)/n
    
    # Compute mu
    #for j in range(K):
    #    denom = np.dot(_post[j,:],Cu_m[:,j])
    #    if denom >= 1:
    #        mu[j,:] = np.matmul(np.multiply(_post[j,:],np.transpose(Cu_m[:,j])),X)/denom
            #mu = np.divide(np.matmul(_post,X),np.transpose(np.tile(np.sum(_post,axis=1),(d,1))))
        

    # Compute sigma**2
    for j in range(K):
        sse, weight = 0,0
        for l in range(d):
            mask = (X[:,l]!=0)
            n_sum = post[mask,j].sum()
            if(n_sum>=1):
                mu[j, l] = (X[mask, l] @ post[mask, j]) / n_sum
            sse += ((mu[j,l]-X[mask,l])**2) @ post[mask,j]
            weight += n_sum
        var[j] = max(sse/weight, min_variance)
            
        #denominateur = 0
        #for i in range(n):
        #    var[j] = var[j] + Cu_m[i,j]*_post[j,i]*np.linalg.norm(np.multiply(X[i,:],Cu_m[i,:])-np.multiply(mu[j,:],Cu_m[i,:]))**2#/(np.sum(np.multiply(_post[j,:],Cu_v)))
        #    denominateur = denominateur + Cu_v[i]*_post[j,i]
        #var[j] = var[j]/denominateur
        #var[j] = max(var[j],min_variance)
            

    
    new_mixture = GaussianMixture(mu,var,p)
    return new_mixture


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_likl = None
    likl = None
    while (prev_likl is None or likl - prev_likl > 1e-6*np.linalg.norm(likl)):
        prev_likl = likl
        post, likl = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, likl


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
