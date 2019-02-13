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
X = np.array(X.values)
y = np.array(y.values)

X_norm, y_norm, mean_X, std_X, mean_y, std_y = normalizar_caracteristica(X, y)

#adicionar uma coluna de 1s a X para poder multiplicar as arrays X e theta
X_norm = np.c_[np.ones((X.shape[0],1)), X_norm]


######


#x = [[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
#x = [[1000,2000,3000],[1,2,3],[1,2,3],[1,2,3]]
#y = [[3,4],[3,4],[3,4],[3,4],[3,4]]
#y = [6,6,6,6]

x = X_norm
y = y_norm

m= len(x)
th = np.array([1,1,1])
alpha = 0.01
epochs = 500

j = custo_reglin_multi(x,y,th)

print(" ")
print("Custo inicial, com theta = ",th," : ", j)
print(" ")

gd = gd(x, y, alpha, epochs, th)

print("Custo final e valores de theta após ",epochs, " epochs:")
print(gd)
print(" ")
