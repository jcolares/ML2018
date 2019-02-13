import numpy as np
from custo_reglin_multi import custo_reglin_multi

def gd(X, y, alpha, epochs, theta):

    m = len(y)
    params = len(theta)
    #av = np.zeros((epochs))

    for i in range(epochs):
        for j in range (params):
            h = x.dot(theta.T)
            dist = np.subtract(h, y)
            theta[j] = theta[j] - alpha / m * np.sum( dist * x[j])    
            #theta[j] = alpha/m * np.sum((h-y)*x[j]) 
            #theta[j] = theta[j] - alpha * np.sum((h-y)*x[j]) / m
            custo = custo_reglin_multi(x, y, theta)
        av[i] = (custo, theta)
        #print(custo[-1], theta)         

    return  (theta, custo, av)

#########3333 teste  apagar daqui para baixo

import os, sys
import pandas as pd
import numpy as np
from normalizacao import normalizar_caracteristica
from numpy.linalg import norm
from custo_reglin_multi import custo_reglin_multi
from gd_reglin_multi import gd

#Carrega dados 
data = pd.read_csv("ex1data2.txt", header=None)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# converte os valores em numpy arrays
X = np.array(X)  #X = np.array(X.values)
y = np.array(y) # y = np.array(y.values)

X_norm, y_norm, mean_X, std_X, mean_y, std_y = normalizar_caracteristica(X, y)

#adicionar uma coluna de 1s a X para poder multiplicar as arrays X e theta
X_norm = np.c_[np.ones((X.shape[0],1)), X_norm]


x = X_norm
y = y_norm

m= len(x)
th = np.matrix([10, 2, 10])
alpha = 1.2
epochs = 100

result = gd(x, y, alpha, epochs, th)
print(result)
print("END")