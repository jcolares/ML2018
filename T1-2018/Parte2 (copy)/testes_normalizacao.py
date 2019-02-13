
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np

from custo_reglin_multi import custo_reglin_multi
from normalizacao import normalizarCaracteristica 
from gd_reglin_multi import gd

#Carrega dados 
data = pd.read_csv("ex1data2.txt", header=None)

#acrescenta uma coluna preenchida com 1s
data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# converte os valores em numpy arrays
X = np.array(X.values)
y = np.array(y.values)

#Testes da função de normalização

result = normalizarCaracteristica(X)
Xnorm = result[0]
Xmean = result[1]
Xstd = result[2]
print(Xnorm) #imprime os dados normalizados
print(Xnorm[:,0]) #imprime os dados normalizados apenas da primeira coluna de X
print(Xmean) #imprime a média de todas as colunas
print(Xstd)
print(Xstd[1]) #imprime o desvio padtão da segunda coluna

