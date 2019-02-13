import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def sigmoide(z):
    return (1.0 / (1.0 + np.exp(-z)))

def costFunctionReg(theta, X, y, lambd):
    # Conversão dos parâmetros da função em arrays
    theta = np.array(theta)
    X = np.array(X)
    y = np.array(y)

    # Quantidade de exemplos (m)
    m = len(X)

    #hipótese (h(x))
    h = sigmoide(np.dot(X, theta))    

    #lado esquerdo da equação do custo
    #custo0 = (-1*y).T.dot(np.log(sigmoide(np.dot(X, theta))))
    custo0 = (-1*y).T.dot(np.log(h))

    #lado direito da equação do custo 
    custo1 = (1-y).T.dot(np.log(1-h)) 

    #termo de regularização 
    reg = (lambd/2) * np.sum(np.dot(theta[1:].T,theta[1:])) 

    #custo regularizado 
    custoReg = (1/m) * ( np.sum(custo0 - custo1) + reg ) 
    return (custoReg)

