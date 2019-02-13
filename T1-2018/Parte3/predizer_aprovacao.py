import numpy as np
from sigmoide import sigmoide

def predizer(theta, X):
    ############################
    #		SEU CÓDIGO AQUI
    #		Essa função deve retornar a classe prevista (1 ou 0) para cada exemplo em X
    ############################
    #calcula a probabilidade com a função sigmoide e os parâmetros recebidos
    probabilidade = sigmoide(X.dot(theta))
    #converte os valores obtidos em 0s e 1s, caso sejam menores ou maiores que 0,5.
    return [1 if x >= 0.5 else 0 for x in probabilidade]


def acuracia(X, y, theta): 
    # obtem as predições (0s e 1s)   
    predicoes = predizer(theta, X) 
    #compara os valores preditos com os valores reais (y)      
    corretas = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predicoes, y)]  
    #calcula o percentual de predições corretas
    acc = (sum(map(int, corretas)) % len(corretas))  
    #imprime o resultado
    print('Acurácia {0}%'.format(acc))
