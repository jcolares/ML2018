
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
from mapFeature import readData
from mapFeature import mapFeature


#carrega os dados do arquivo
data = pd.read_csv('ex2data2.txt', header = None ,names=['Test 1', 'Test 2', 'Resultado'])

#separa o dataset em dois: X com duas colunas e y com uma
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]


# Lê os dados do arquivo 
X, y = readData()

#Separa os dados de X de acordo com o valor de y
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


#print(mapfeature2(X))
A  = mapFeature(X)
print(X)