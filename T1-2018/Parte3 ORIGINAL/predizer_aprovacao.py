import numpy as np
from sigmoide import sigmoide

def predizer(theta, X):
    ############################
    #		SEU CÓDIGO AQUI
    #		Essa função deve retornar a classe prevista (1 ou 0) para cada exemplo em X
    ############################
    probabilidade = sigmoide(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probabilidade]

def acuracia(X, theta, result):
    theta_min = np.matrix(result[0])  
    predicoes = predizer(theta_min, X)  
    corretas = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predicoes, y)]  
    acc = (sum(map(int, corretas)) % len(corretas))  
    print('Accurácia {0}%'.format(acc))
