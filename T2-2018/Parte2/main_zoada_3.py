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
	m = len(X)
	#X = np.matrix(X)
	sigma = X.T.dot(X) / m
	# Decomposicao SVD da matriz sigma: 
	U, S, V = np.linalg.svd(sigma)
	return (U, S)

def project_data(X, U, K):
	# seleciona apenas as primeiras K colunas de U:
	U_reduce = U[:, 0:K]
	# Inizializa Z com o comprimento igual a X, mas com apenas K colunas: 
	Z = np.zeros((len(X), K))
	# Multiplica cada linha de X por U_reduce
	Z = X.dot(U_reduce)	
	print("Valor do primeiro exemplo de dados ")
	print("após a redução de dimensões:")
	print(Z[0])
	# Retorna a nova matriz com dimensões reduzidas Z:
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
	raw_mat = scipy.io.loadmat("../data/ex7data1.mat")
	X = raw_mat.get("X")
	#X = np.matrix(X)
	plt.cla()
	plt.plot(X[:,0], X[:,1], 'bo')
	plt.show()

	X_norm, mu, sigma = normalize_features(X)
	#mu = np.matrix(mu).T
	#sigma = np.matrix(sigma).T

	U, S = pca(X_norm)

	plt.cla()
	plt.axis('equal')
	plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
	
	K = 2
	for axis, color in zip(U[:K], ["yellow","green"]):
		start, end = np.zeros(2), (mu + sigma * axis)[:K] - (mu)[:K]
		plt.annotate('', xy=end,xytext=start, arrowprops=dict(facecolor=color, width=1.0))
	plt.axis('equal')
	plt.show()
	'''
	print("mu: ", np.shape(mu), mu)
	print("sigma: ", np.shape(sigma), sigma)
	print("axis: ", np.shape(axis))
	print(axis)
	'''

	'''
	K = 1
	Z = project_data(X_norm, U, K)
	X_rec = recover_data(Z, U, K)

	plt.cla()
	plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
	plt.plot(X_rec[:,0], X_rec[:,1], 'rx')
	plt.axis('equal')
	plt.show()
	
	explain_variance(S)
	'''
if __name__ == "__main__":
	main()
