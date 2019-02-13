import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib.mlab as mlab
from scipy.io import loadmat  
from scipy import stats 

def estimate_gaussian_params(X):  
	########################
	# SEU CÓDIGO AQUI : 
	# Essa função deve computar e retornar mu e sigma2, 
	# vetores que contêm a média e a variância de cada 
	# característica (feature) de X.
	########################
	# Inicializa a variável feat com o número de dimensões (colunas) em X:
	feat = np.size(X,1)
	# Inicializa m com o número de exemplos (linhas) existentes em X:
	m = len(X)
	# Inicializa a array sigma2 com o mesmo compromento de X: 
	sigma2 = np.zeros(feat)
	# Inicializa a array mu, com a mesm largura (colunas) de X:
	mu = np.zeros(feat)
	for i in range(feat):
		# Calcula a média da coluna
		mu[i] = np.sum(X[:,i]) / m
		# Calcula a variância da coluna
		sigma2[i] = sum( np.power(X[:,i]-mu[i],2) )/m
	# retorna os valores calculados
	return (mu, sigma2)

def f1(preds, pval ,yval):
    # inicializa as variaveis 
    # TP - true positives
    tp = 0
    # FP - false positives
    fp = 0
	# FN - false negatives
    fn = 0 
    # Executa os comandos do loop para cada uma das linhas do conjunto de validacao cruzada:
    for i in range(len(pval)):
        # Faz a contagem de TPs, FPs e FNs
        if preds[i] == True and int(yval[i]) == 1:
            tp = tp + 1
        if preds[i] == True and int(yval[i]) == 0:
            fp = fp + 1
        if preds[i] == True and int(yval[i]) == 1:
            fn = fn + 1
    # Faz os calculos da precisao, revocacao e do Fscore:
    if tp + fp > 0: 
        prec = tp / (tp + fp)
    else: 
        prec = 0
    if tp + fn > 0: 
        rec = tp / (tp + fn)
    else:
        rec = 0
    if prec + rec > 0:	
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0
    return f1

def select_epsilon(pval, yval):  
    best_epsilon_value = 0
    best_f1_value = 0
    step_size = (pval.max() - pval.min()) / 1000
    novof1 = 0
    print('step size: ' + str(step_size))
    for epsilon in np.arange(pval.min(), pval.max(), step_size):		
        # armazena flag true quando o valor predito é menor que epsilon
        preds = pval < epsilon
		########################
		# SEU CÓDIGO AQUI : 
		# Dentro deste loop, você deve implementar lógica para 
		# definir corretamente os valores das variáveis 
		# best_epsilon_value e best_f1_value.
		########################
		# chama a função f1 para calcular o valor de F1
        novof1 = f1(preds,pval, yval)
		# Se encontrado um valor melhor de FI, armazena.
        if novof1 > best_f1_value:
            best_f1_value = novof1
            best_epsilon_value = epsilon
    return best_epsilon_value, best_f1_value

def main():
	data = loadmat('../data/ex8data1.mat') 
	X = data['X']  

	(mu, sigma2) = estimate_gaussian_params(X)
	print('mu: ' , mu)
	print('variance: ' , sigma2)

	# Plot dataset
	plt.scatter(X[:,0], X[:,1], marker='x')  
	plt.axis('equal')
	plt.show()

	# Plot dataset and contour lines
	plt.scatter(X[:,0], X[:,1], marker='x')  
	x = np.arange(0, 25, .025)
	y = np.arange(0, 25, .025)
	first_axis, second_axis = np.meshgrid(x, y)
	Z = mlab.bivariate_normal(first_axis, second_axis, np.sqrt(sigma2[0]), np.sqrt(sigma2[1]), mu[0], mu[1])
	plt.contour(first_axis, second_axis, Z, 10, cmap=plt.cm.jet)
	plt.axis('equal')
	plt.show()

	# Load validation dataset
	Xval = data['Xval']  
	yval = data['yval'].flatten()

	# array armzena o valor do desvio padrão das colunas de X a p
	stddev = np.sqrt(sigma2)

	# Calcula a desnsidade de probabilidade para cada coluna de X e armazena em pval
	pval = np.zeros((Xval.shape[0], Xval.shape[1]))  
	pval[:,0] = stats.norm.pdf(Xval[:,0], mu[0], stddev[0])  
	pval[:,1] = stats.norm.pdf(Xval[:,1], mu[1], stddev[1])  
	print(np.prod(pval, axis=1).shape)
    
	# Chama a função select_epsilon(), armazena o resultado e imprime
	epsilon, _ = select_epsilon(np.prod(pval, axis=1), yval)  
	print('Best value found for epsilon: ' + str(epsilon))

	# Computando a densidade de probabilidade 
	# de cada um dos valores do dataset em 
	# relação a distribuição gaussiana
	p = np.zeros((X.shape[0], X.shape[1]))  
	p[:,0] = stats.norm.pdf(X[:,0], mu[0], stddev[0])  
	p[:,1] = stats.norm.pdf(X[:,1], mu[1], stddev[1])

	# Apply model to detect abnormal examples in X
	anomalies = np.where(np.prod(p, axis=1) < epsilon)

	# Plot the dataset X again, this time highlighting the abnormal examples.
	plt.clf()
	plt.scatter(X[:,0], X[:,1], marker='x')  
	plt.scatter(X[anomalies[0],0], X[anomalies[0],1], s=50, color='r', marker='x')  
	plt.axis('equal')
	plt.show()

if __name__ == "__main__":
	main()