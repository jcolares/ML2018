import numpy as np

def custo_reglin_multi(X, y, theta):

    # Quantidade de exemplos
    m = len(X)

    # Computa a função de custo J
    #J = (np.sum( np.power( (X.dot(theta.T)- y) , 2) ))/ (2 * m)
    J = (np.sum( np.power( np.subtract(X.dot(theta.T), y) , 2) ))/ (2 * m)

    return J

