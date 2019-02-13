import numpy as np
from sigmoide import sigmoide

def normalizar_caracteristicas(X, y):
    X_mean = np.average(X, axis = 0) # média de X de cada coluna de X
    y_mean = np.average(y) # média de y
    X_stdev = np.std(X, axis = 0) # desvio padrão de cada coluna de X
    y_stdev = np.std(y)  #desvio padrao de y
    X_norm = (X - X_mean) / X_stdev  # Normalização z-score de X
    y_norm = (y - y_mean) / y_stdev  # Normalização z-score de y
    return X_norm, X_mean, X_stdev, y_norm, y_mean, y_stdev  #retorno da função
    
def custo_reglog(theta, X, y):
    # Conversão dos parâmetros da função em matrizes
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    # Cálculo dp custo     
    grad0 = np.multiply(-1 * y, np.log(sigmoide(X * theta.T)))
    grad1 = np.multiply((1 - y), np.log(1.00 - sigmoide(X * theta.T)))
    return np.sum(grad0 - grad1) / (len(X))

