import numpy as np
from custo_reglin_multi import custo_reglin_multi

def gd(X, y, alpha, epochs, theta):

    m = len(y)
    params = len(theta)
    for i in range(epochs):
        for j in range (params):
            h = X.dot(theta.T)
            dist = np.subtract(h, y)
            theta[j] = theta[j] - alpha / m * np.sum( dist * X[j])    
            custo = custo_reglin_multi(X, y, theta)    

    return  (theta, custo)
