import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.io

def normalize_features(X):
	mu = np.mean(X,axis=0)
	sigma = np.std(X,axis=0)
	normalized_X = np.divide(X - mu,sigma)
	return (normalized_X, mu, sigma)

def pca(X):
	########################
	# SEU CÓDIGO AQUI : 
	# Essa função deve retornar U e S, duas das
	# três matrizes geradas pela decomposição 
	# da matriz de covariância de X.
	########################
	# Calculo da matriz de covariancia sigma:
	X = np.matrix(X)
	m = len(X)
	sigma = X * X.T / m
	# Decomposição SVD da matriz sigma: 
	U, S, V = np.linalg.svd(sigma)
	return (U, S)

def project_data(X, U, K):
	# Cruia uma nova matriz contendo apenas as primeiras K colunas de X:
	U_reduce = U[:, 0:K]
	# Cria matriz Z, com o comprimento de X, mas com K colunas:
	Z = np.zeros((len(X), K))
	# Para cada linha de X, executa: 
	for i in range(len(X)):
		x = X[i,:]
		projection_k = np.dot(x, U_reduce)
		Z[i] = projection_k
	return Z

def recover_data(Z, U, K):
	X_rec = np.zeros((len(Z), len(U)))
	for i in range(len(Z)):
		v = Z[i,:]
		for j in range(np.size(U,1)):
			recovered_j = np.dot(v.T,U[j,0:K])
			X_rec[i][j] = recovered_j
	return X_rec

def explain_variance(S):
	########################
	### SEU CÓDIGO AQUI  ###
	########################

	# implement code to print the percentages 
	# of variation for each dimension.
	pass

def main():
	# obtém os dados no arquivo e armazena na array X:
	raw_mat = scipy.io.loadmat("../data/ex7data1.mat")
	X = raw_mat.get("X")
	# Plota os dados de X nem um scatterplot:
	plt.cla()
	plt.plot(X[:,0], X[:,1], 'bo')
	plt.show()

	# Chama a função que normaliza os valores de X e armazena em Xnorm
	# também armazena nas variaveis mu e sigma os valores de media e desvio padrão
	# para posterior reconstrução dos dados:
	X_norm, mu, sigma = normalize_features(X)
	
	# Chama a função PCA para decompor X e obter as matrizes U e S, 
	# que serão usadas para reduzir o numero de dimensoes de X:
	U, S = pca(X_norm)

	# Plota um gráfico de dispersão com os valores de X normalizados
	plt.cla()
	plt.axis('equal')
	plt.plot(X_norm[:,0], X_norm[:,1], 'bo')

	K = 2
	mu = np.matrix(mu).T
	sigma = np.matrix(sigma).T
	for axis, color in zip(U[:K], ["yellow","green"]):
		start, end = np.zeros(2), (mu + sigma * axis)[:K] - (mu)[:K]
		plt.annotate('', xy=end,xytext=start, arrowprops=dict(facecolor=color, width=1.0))
	plt.axis('equal')
	plt.show()


	K = 1
	Z = project_data(X_norm, U, K)
	X_rec = recover_data(Z, U, K)

	plt.cla()
	plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
	plt.plot(X_rec[:,0], X_rec[:,1], 'rx')
	plt.axis('equal')
	plt.show()

	explain_variance(S)

if __name__ == "__main__":
	main()
