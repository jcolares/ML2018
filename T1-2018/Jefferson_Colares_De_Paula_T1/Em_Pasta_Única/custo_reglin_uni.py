#coding=ISO8859-1
import numpy as np

def custo_reglin_uni(X, y, theta):

    # Quantidade de exemplos de treinamento
    m = len(y)

    # Computar a função do custo J
    J = (np.sum((X.dot(theta) - y)**2)) / (2 * m)

    return J



def custo_reglin(X, y, alpha, epochs, theta = np.array([0,0], ndmin = 2).T):

    m = len(y)

    cost = np.zeros(epochs)

    for i in range(epochs):

        h = X.dot(theta)

        loss = h - y

        gradient = X.T.dot(loss) / m

        theta = theta - (alpha * gradient)

        cost[i] = custo_reglin_uni(X, y, theta = theta)

    return cost[-1], theta
