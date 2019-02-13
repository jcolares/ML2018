#declaração de encoding para possibilitar a utilização desse arquivo vom o Linux

import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np

#from custo_reglin_uni import custo_reglin_uni
from custo_reglin_uni import custo_reglin

#modificação no filepath para possibilitar a utilização desse arquivo com o Linux
#filepath = "\ex1data1.txt"
filepath = "/ex1data1.txt"

def importarDados(filepath,names):
    path = os.getcwd() + filepath
    data = pd.read_csv(path, header=None, names=names)

    # adiciona uma coluna de 1s referente a variavel x0
    data.insert(0, 'Ones', 1)
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

#Versão original abaixo tentava passar as duas colunas de X e provocava erro na geração do gráfico
#plt.scatter(X, y, color='blue', marker='x')
plt.scatter(X[:,1], y, color='blue', marker='x')
plt.title('População da cidade x Lucro da filial')
plt.xlabel('População da cidade (10k)')
plt.ylabel('Lucro (10k)')
plt.savefig('plot1.1.png')
plt.show()
