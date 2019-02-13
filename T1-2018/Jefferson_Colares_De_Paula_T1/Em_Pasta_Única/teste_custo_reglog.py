#coding=ISO8859-1
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from custo_reglog import custo_reglog 
from custo_reglog import normalizar_caracteristicas

# Carregando os dados do dataset e armazendo em um array. Em seguida damos uma r?pida visualizada nos dados
data = pd.read_csv('ex2data1.txt', header=None, names=['Prova 1', 'Prova 2', 'Aprovado'])  

#Separa as colunas do dataset em Características e Rótulos
X = data.iloc[:,0:2]  
y = data.iloc[:,2:3]

#Normaliza as características 
X_norm, X_mean, X_stdev, y_norm, y_mean, y_stdev  = normalizar_caracteristicas(X, y)

# Insere coluna de 1’s no conjunto de dados X
X_norm = np.c_[np.ones((X.shape[0],1)), X_norm]

# Inicia o vetor de parâmetros theta
theta = np.array([0, 0, 0],ndmin=2)

# Computa o valor da função de custo J
J = custo_reglog(theta, X_norm, y)

#Exibe o resultado do teste da função de custo. 
#O resultado esperado é aproximadamente 0,693, quando theta = [0, 0, 0]
print(J)

