
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from custo_reglin_multi import custo_reglin_multi
from normalizacao import normalizarCaracteristica 
from gd_reglin_multi import gd

#Carrega dados 
data = pd.read_csv("ex1data2.txt", header=None)

#acrescenta uma coluna preenchida com 1s
#data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# converte os valores em numpy arrays
X = np.array(X.values)
y = np.array(y.values)

#Obtém X e y normalizados
all_X = normalizarCaracteristica(X)
all_y = normalizarCaracteristica(y)
X_norm = all_X[0]
y_norm = all_y[0]

#X_norm, y_norm, mean_X, std_X, mean_y, std_y = normalizar_caracteristica(X, y)

#adicionar uma coluna de 1s a X para poder multiplicar as arrays X e theta
X_norm = np.c_[np.ones((X.shape[0],1)), X_norm]

#Definição de parâmetros
theta = np.matrix([10, 2, 10])
alpha = 1
epochs = 15

# Cálculo do custo (inicial)
j = custo_reglin_multi(X_norm, y_norm, theta)
print(" ")
print("Custo inicial, com theta = ",theta," : ", j)
print(" ")

gd = gd(X_norm, y_norm, alpha, epochs, theta)

print("Custo final : ", gd[1], "  Theta final : ", gd[0]) 
print(" ")
