import numpy as np
from scipy.optimize import fmin_cg


def cofi_cost_func(params, Y, R, num_users, num_movies, num_features, Lambda):

    # Obtém as matrizes X e Theta a partir dos params
    X = np.array(params[:num_movies*num_features]).reshape(num_features, num_movies).T.copy()
    Theta = np.array(params[num_movies*num_features:]).reshape(num_features, num_users).T.copy()


    # Você deve retornar os seguintes valores corretamente
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # ====================== SEU CÓDIGO AQUI ======================
    # Instruções: calcular a função de custo regularizada e gradiente  
    # para a filtragem colaborativa. Concretamente, você deve primeiro
    # implementar a função de custo. Depois disso, você deve implementar o
    # gradiente. 
    #
    # Notas: 
    # X - num_movies x num_features: matriz das características dos filmes
    # Theta - num_users x num_features: matriz das características dos usuários
    # Y - num_movies x num_users: matriz de classificações de filmes por usuários
    # R - num_movies x num_users: matriz, onde R (i, j) = 1 se o i-ésimo filme 
    #       foi avaliado pelo j-ésimo usuário
    #
    # Você deve definir as seguintes variáveis ​​corretamente:
    #
    # X_grad - num_movies x num_features matrix, contendo as
    #   derivadas parciais com relação a cada elemento de X
    # Theta_grad - num_users x num_features: matriz, contendo as
    #   derivadas parciais com relação a cada elemento de Theta
    # =============================================================
    '''
    cost = 0
    cost_sq = 0
    dist = np.dot(X, Theta.T) 
    for i in range(num_movies):
        for j in range(num_users):  
            if R[i,j] == True:
                cost_sq = cost_sq + (dist[i,j] - Y[i,j]) ** 2
                cost = cost + (dist[i,j] - Y[i,j])
    J = cost_sq / 2
    
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    for k in range(num_features):
        for i in range(num_movies):
            X_grad[i,k] = X[i,k] * cost
        for j in range(num_users):
            Theta_grad[j,k] = Theta[j,k] * cost
    '''
    #Cálculo do custo (J)
    cost_sq = np.power((np.dot(X, Theta.T) - Y), 2)
    cost_sq_rated = np.multiply(cost_sq,  R)
    J = np.sum(cost_sq_rated) / 2

    #Parcela de regularização do custo
    reg =  Lambda/2 * (np.sum(np.power(Theta, 2)) + np.sum(np.power(X, 2)))

    #Custo regularizado 
    J = J + reg


    #Cálculo dos gradientes 
    cost = np.dot(X, Theta.T) - Y
    cost_rated = np.multiply(cost, R) 
    #Theta_grad = np.dot(cost_rated.T, X) 
    #X_grad = np.dot(cost_rated, Theta)
    
    # Cálculo dos gradientes com regularização
    Theta_grad = np.dot(cost_rated.T, X) + np.dot(Theta, Lambda)
    X_grad = np.dot(cost_rated, Theta) + np.dot(X, Lambda)   

    grad = np.hstack((X_grad.T.flatten(),Theta_grad.T.flatten()))

    return J, grad

