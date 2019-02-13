import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg


#carrega os dados do arquivo
dados = pd.read_csv('ex2data2.txt', header = None ,names=['Test 1', 'Test 2', 'Resultado'])

#separa o dataset em dois: X com duas colunas e y com uma
cols = dados.shape[1]
X = dados.iloc[:,0:cols-1]
y = dados.iloc[:,cols-1:cols]

#converte os datasets em arrays
X = np.array(X.values)
y = np.array(y.values)

#Inicializa parâmetros da função de custo
Xmap = mapFeature(X[:,0],X[:,1])
thetaInicial = np.ones(28)


print(' ')
print('VARIAÇÃO DO CUSTO OBTIDO COM DIFERENTES VALORES DE LAMBDA:')
print(' ')
lambd = 0
print('Custo com lambda = 0: ', costFunctionReg(thetaInicial, Xmap, y, lambd))
print(' ')
lambd = 1
print('Custo com lambda = 1: ', costFunctionReg(thetaInicial, Xmap, y, lambd))
print(' ')
lambd = 100
print('Custo com lambda = 110: ', costFunctionReg(thetaInicial, Xmap, y, lambd))
print(' ')
