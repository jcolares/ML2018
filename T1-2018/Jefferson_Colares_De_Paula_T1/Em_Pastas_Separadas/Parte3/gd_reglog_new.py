import numpy as np
from custo_reglog import custo_reglog
from sigmoide import sigmoide

def gd_reglog(theta, X, y, epochs, alpha):
    #Converte theta, X e y em matrizes
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    #Normaliza os valores de X (Z-score):
    X = (X - np.average(X))/np.std(X)  

    #parametros = int(theta.ravel().shape[1])
    
    grad = np.zeros(parametros)

    erro = sigmoide(X * theta.T) - y
    
    for i in range(parametros):
        term = np.multiply(erro, X[:,i])  
        grad[i] = np.sum(term) / len(X)
        custo = custo_reglog(grad, X, y)
        
    return grad, custo