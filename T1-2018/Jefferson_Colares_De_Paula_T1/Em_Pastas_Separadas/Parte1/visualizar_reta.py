#coding=ISO8859-1
#declaração de encoding para possibilitar a utilização desse arquivo vom o Linux

import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
from custo_reglin_uni import custo_reglin

#carrega os dados do arquivo
dataset = pd.read_csv('ex1data1.txt', header = None)

#acrescenta uma coluna preenchida com 1s
dataset.insert(0, 'Ones', 1)

#separa o dataset em dois: X com duas colunas e y com uma
cols = dataset.shape[1]
X = dataset.iloc[:,0:cols-1]
y = dataset.iloc[:,cols-1:cols]

#converte os datasets em arrays
X = np.array(X.values)
y = np.array(y.values)

#utiliza o GD para obter a reta correta da regressão
custo, theta = custo_reglin(X, y, 0.01, 5000)
print("Custo: ",custo)
print("Theta: ",theta.T)

#imprime o gráfico da regressão linear
t = np.arange(0, 25, 1)
plt.scatter(X[:,1], y, color='red', marker='x', label='Training Data')
plt.plot(t, theta[0] + (theta[1]*t), color='blue', label='Linear Regression')
plt.axis([4, 25, -5, 25])
plt.title('População da cidade x Lucro da filial')
plt.xlabel('População da cidade (10k)')
plt.ylabel('Lucro (10k)')
plt.legend()
plt.show()

#Prediz os valores do lucro para cidades de 35 e 70 mil habitantes
print("Predição de lucro para cidades:")
print("com 35.000 habitantes: ", theta[0] + theta[1] * 35000)
print("com 70.000 habitantes: ", theta[0] + theta[1] * 70000)
