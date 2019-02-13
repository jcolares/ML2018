#coding=ISO8859-1
#declaração de encoding para possibilitar a utilização desse arquivo vom o Linux

#import matplotlib.pyplot as plt

#chamadas a bibliotecas {
import os, sys
#import pandas as pd
import numpy as np
from custo_reglin_uni import custo_reglin_uni
from gd_reglin_uni import gd_reglin_uni
# }
'''
#modificação no filepath para possibilitar a utilização desse arquivo com o Linux
#filepath = "\ex1data1.txt"
filepath = "/ex1data1.txt"

def importarDados(filepath,names):
    path = os.getcwd() + filepath
    data = pd.read_csv(path, header=None, names=names)

    # adiciona uma coluna de 1s referente a variavel x0
    #data.insert(0, 'Ones', 1)
    # a linha acima fazia parte do código original, mas foi removida para possibilitar a plotagem do gráfico.

    # separa os conjuntos de dados x (caracteristicas) e y (alvo)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    # converte os valores em numpy arrays
    X = np.array(X.values)
    y = np.array(y.values)

    return X,y

X,y = importarDados(filepath,["Population","Profit"])

plt.scatter(X, y, color='blue', marker='x')
plt.title('População da cidade x Lucro da filial')
plt.xlabel('População da cidade (10k)')
plt.ylabel('Lucro (10k)')
plt.savefig('plot1.1.png')
plt.show()


dadosX = [1,2,3,4,5]
dadosY = [2,4,6,8,10]
X = np.array(dadosX)
Y = np.array(dadosY)
theta = 1
print(custo_reglin_uni(X,Y,theta))
print(X.T)
print(X.T)
theta = np.array([0,0], ndmin = 2).T
print(theta)
print(theta.T)

epochs = 5
cost = np.zeros(epochs)
print(cost)

theta = [1,1]
gdd = gd_reglin_uni(X, y, alpha, epochs, theta = np.array([0,0], ndmin = 2).T)
'''
import numpy as np
x = np.array([[10,20,30], [40,50,60]])
y = np.array([[100], [200]])
'''print(x)
print(np.append(y, x, axis=1))
print(x)
'''
z= np.zeros(len(x)).T
print(z)
print(np.append(z, x, axis=1))
print(x)
