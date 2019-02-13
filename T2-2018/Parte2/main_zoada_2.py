import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.io

def normalize_features(X):
	mu = np.mean(X,axis=0)
	sigma = np.std(X,axis=0)
	normalized_X = np.divide(X - mu,sigma)
	normalized_X = np.matrix(normalized_X)
	return (normalized_X, mu, sigma)

def pca(X):
	########################
	# SEU CÓDIGO AQUI : 
	# Essa função deve retornar U e S, duas das
	# três matrizes geradas pela decomposição 
	# da matriz de covariância de X.
	########################
	# Calculo da matriz de covariancia sigma:
	m = len(X)
	#minha formula: sigma = X.dot(X.T) / m
	# Formula do Andrew Ng:
	sigma = X.T.dot(X) / m
	# Decomposição SVD da matriz sigma: 
	U, S, V = np.linalg.svd(sigma)
	U = np.matrix(U)
	S = np.matrix(S)
	print("U:", np.shape(U))
	print(U)
	return (U, S)

def project_data(X, U, K):
	U_reduce = np.matrix(U[:,0:K])
	###### TESTES
	print("U_reduce: ")
	print(np.shape(U_reduce))
	x = X[:,0]
	#Eu: Z = U_reduce.T.dot(x)
	#ANg
	Z = U_reduce.T.dot(X)
	print("x: ")
	print(np.shape(X))
	print("Z: ")
	print(np.shape(Z))
	print(Z)
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
	# Lê dados do arquivo e armazena em X:
	raw_mat = scipy.io.loadmat("../data/ex7data1.mat")
	X = raw_mat.get("X")
	# Converte X para matriz:
	X = np.matrix(X)
	#print("x1:", np.shape(X[:,0]))
	#print(X[:,0])
	#print()
	# Plota um gráfico de dispersão com os dados de X:
	plt.cla()
	plt.plot(X[:,0], X[:,1], 'bo')
	# temp plt.show()

	# Chama a função que retorna X normalizado, a média(mu) e o desvio padrão(sigma):
	X_norm, mu, sigma = normalize_features(X)
	# Chama a função que decompõe X usando SVD e armazena o resultado em U e S:
	U, S = pca(X_norm)

	K = 1

	###### TESTES
	print("X: ")
	print(np.shape(X))
	print("U: ")
	print(np.shape(U))
	print("K: ", K)
	print(U[:,0:K])
	'''
	plt.cla()
	plt.axis('equal')
	plt.plot(X_norm[:,0], X_norm[:,1], 'bo')

	K = 1
	for axis, color in zip(U[:K], ["yellow","green"]):
		start, end = np.zeros(2), (mu + sigma * axis)[:K] - (mu)[:K]
		plt.annotate('', xy=end,xytext=start, arrowprops=dict(facecolor=color, width=1.0))
	plt.axis('equal')
	plt.show()
	'''
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
