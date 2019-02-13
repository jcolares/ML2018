import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
from mapFeature import mapFeature

def plotDecisionBoundary(X, y, data, theta):

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

    #Gera dados para plotar a fronteira de decisão
    xs = np.linspace(-1,1.5,50)
    ys = np.linspace(-1,1.5,50)
    zs = np.zeros((len(xs),len(ys)))
    #meu código (funciona)
    #for i in range(len(xs)):
    #    for j in range(len(ys)):
    #        zs[i][j] = np.dot( mapFeature(np.array([xs[i]]),np.array([ys[j]])) , theta )
    #zs = zs.transpose()
    for i in range(len(xs)):
        for j in range(len(ys)):
            myfeaturesij = mapFeature(np.array([xs[i]]),np.array([ys[j]]))
            zs[i][j] = np.dot(theta,myfeaturesij.T)
    zs = zs.transpose()
    print(zs)
    #Plota a fronteira de decisaão
    plt.contour( xs, ys, zs, [0])
    fig.savefig('plot4.3.png')

    plt.show()

    return()
