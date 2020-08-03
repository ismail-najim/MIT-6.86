import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)[]
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return np.power( np.matmul(X,np.transpose(Y)) + c ,p )


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    n_rows = X.shape[0]
    n_cols = Y.shape[0]
    RBF = np.zeros((n_rows,n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            RBF[i,j] = np.linalg.norm(X[i,:])**2+np.linalg.norm(Y[j,:])**2-2*np.dot(X[i,:],Y[j,:])
    
    return np.exp(-gamma*(RBF))
