"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    likl = np.zeros((n, K))
    
    # Calculate likelihood for each point
    for i in range(n):
        for j in range(K):
            likl[i,j]= mixture.p[j]*(1/(2*np.pi*mixture.var[j])**(d/2))*np.exp(-1/2*(1/mixture.var[j])*np.linalg.norm(X[i]-mixture.mu[j])**2)
        post[i,:]=likl[i,:]/np.sum(likl[i,:])
    
    LL = 0
    
    for i in range(n):
        LL = LL + np.log(np.sum(likl[i,:]))
    return post, LL

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # Define parameters and dimensions
    n, d = X.shape
    _, K = post.shape
    mu = np.zeros((K,d))
    p = np.zeros(K)
    var = np.zeros(K)
    _post = np.copy(np.transpose(post))
    
    # Compute p
    p = np.sum(_post,axis=1)/n
    mu = np.divide(np.matmul(_post,X),np.transpose(np.tile(np.sum(_post,axis=1),(d,1))))

    
    for j in range(K):
        for i in range(n):
            var[j] = var[j] + _post[j,i]*np.linalg.norm(X[i,:]-mu[j,:])**2/(d*np.sum(_post[j,:]))

    
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
