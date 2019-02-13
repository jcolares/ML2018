
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
from mapFeature import readData
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from costFunctionReg import gd_reglog_reg  
#from costFunctionReg import grad_reglog   #remover

# Lê os dados do arquivo 
X, y, data = readData()

#Separa os dados obtidos do arquivo em 2 conjuntos (dataframes), 
# de acordo com o conteúdo da coluna "Resultado".  
positivo = data[data['Resultado'].isin([1])]  
negativo = data[data['Resultado'].isin([0])]

#Plota o gráfico de dispersão conforme o item 4.1 do enunciado.
fig = plt.figure()
ax = fig.subplots()  
ax.axis([-1, 1.5, -1, 1.5])
ax.scatter(positivo['Test 1'], positivo['Test 2'], s=50, c='k', marker='+', label='y=1')  
ax.scatter(negativo['Test 1'], negativo['Test 2'], s=50, c='y', marker='o', label='y=0')  
ax.legend()  
ax.set_xlabel('Microchip Test 1')  
ax.set_ylabel('Microchip Test 2')  
fig.savefig('plot4.1.png')
fig.show()

#Inicializa a array de parâmetros theta
theta = np.zeros(28)

#Inicializa o hiperparâmetro Lambda
lbd = 0

#Engenharia de parâmetros: chama a função mapFeature, 
# que aumenta as características (X) para 28 colunas
X = mapFeature(X)

#Teste da função do custo. Utilzando theta = 0 e lambda = 0
theta = np.zeros(28)
lbd = 0
print("Custo (com theta=0): ", end="" )
print(costFunctionReg(X, y, lbd, theta))
print("Valor esperado: 0.693")




#TESTES
theta = np.zeros(28)
lbd = 0

#print(grad_reglog(X, y, theta))

theta_otim = gd_reglog_reg(X, y, theta, lbd)
print(theta_otim)

