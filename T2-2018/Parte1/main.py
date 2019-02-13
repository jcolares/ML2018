import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.io
import matplotlib.pyplot as plt

def distancia(x, centroide):
	dx = x[0] - centroide[0]
	dy = x[1] - centroide[1]
	d = np.sqrt(dx **2 + dy **2)
	return(d)

def find_closest_centroids(X, centroids):
	K = np.size(centroids, 1)
	idx = np.zeros((len(X), 1), dtype=np.int8)
	##################################################
	# SEU CÓDIGO AQUI
	K = len(centroids)
	dist = np.zeros((len(X), 1))
	for i in range(len(idx)):
		#obtém a distancia entre o ponto e o primeiro centroide:
		dist[i] = dist_nova = distancia(X[i],centroids[0]) 
		idx[i] = 0
		#executa os passos abaixo para os K centroides:
		for j in range(K): 
			#calcula a distancia entre o ponto e o proximo centroide: 
			dist_nova = distancia(X[i],centroids[j]) 
			#se a nova distancia obtida for menor que a atual
			if dist_nova < dist[i]: 
				#armazena a nova menor distancia:
				dist[i] = dist_nova 
				#armazena o id do centroide atual em idx			 
				idx[i] = j                           
	##################################################
	return idx

def compute_centroids(X, idx, K):
	centroids = np.zeros((K,np.size(X,1)))
	##################################################
	# SEU CÓDIGO AQUI
	for i in range(K):
		#concatena a atribuição de clusters e X
		idxX = np.concatenate( (idx ,X ), axis = 1)
		#cria uma array booleana que identifica os elementos de X que pertencem ao cluster k
		ind = (idxX[:,0] == i)
		#cria uma array contendo apenas os elementos de X que pertencem ao cluster k
		C = (X[ind])
		#calcula a média de x e y
		meanx = np.mean(C[:,0], axis =0 )
		meany = np.mean(C[:,1], axis =0 )
		# atribui as médias obtidas às coordenadas do centroide
		centroids[i] = np.array([meanx, meany])
	##################################################
	return centroids

def kmeans_init_centroids(X, K):
	return X[np.random.choice(X.shape[0], K, replace=False)]

def run_kmeans(X, initial_centroids, max_iters, plot_progress=False):
	K = np.size(initial_centroids, 0)
	centroids = initial_centroids 
	previous_centroids = centroids

	for iter in range(max_iters):
		# Assignment of examples do centroids
		idx = find_closest_centroids(X, centroids)

		# PLot the evolution in centroids through the iterations
		if plot_progress:
			plt.scatter(X[np.where(idx==0),0],X[np.where(idx==0),1], marker='x')
			plt.scatter(X[np.where(idx==1),0],X[np.where(idx==1),1], marker='x')
			plt.scatter(X[np.where(idx==2),0],X[np.where(idx==2),1], marker='x')
			plt.plot(previous_centroids[:,0], previous_centroids[:,1], 'yo')
			plt.plot(centroids[:,0], centroids[:,1], 'bo')
			plt.show()

		previous_centroids = centroids

		# Compute new centroids
		centroids = compute_centroids(X, idx, K)

	return (centroids, idx)

def main():
	# Find closest centroids
	raw_mat = scipy.io.loadmat("../data/ex7data2.mat")
	X = raw_mat.get("X")

	#Numero de centroides
	K = 3	
	'''
	# Fixed seeds (i.e., initial centroids)
	initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
	idx = find_closest_centroids(X, initial_centroids)
	
	# Plot initial assignments.
	plt.scatter(X[np.where(idx==0),0],X[np.where(idx==0),1], marker='x')
	plt.scatter(X[np.where(idx==1),0],X[np.where(idx==1),1], marker='x')
	plt.scatter(X[np.where(idx==2),0],X[np.where(idx==2),1], marker='x')
	plt.title('Initial assignments')
	plt.show()
	
	print('Cluster assignments for the first, second and third examples: ' + str(idx[0:3].flatten()))

	# Compute initial means
	# centroids = compute_centroids(X, idx, K)

	# Now run 10 iterations of K-means on fixed seeds
	initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
	#initial_centroids = kmeans_init_centroids(X, K)
	max_iters = 10
	centroids, idx = run_kmeans(X, initial_centroids, max_iters, plot_progress=False)
	print('Centroids after the 1st update:\n' + str(centroids))
	
	# Plot final clustering.
	plt.scatter(X[np.where(idx==0),0],X[np.where(idx==0),1], marker='x')
	plt.scatter(X[np.where(idx==1),0],X[np.where(idx==1),1], marker='x')
	plt.scatter(X[np.where(idx==2),0],X[np.where(idx==2),1], marker='x')
	plt.title('Final clustering')
	plt.show()
	'''
	#####################################################################
	# SEU CÓDIGO AQUI: repita a executação acima, 
	# desta vez iniciando o centróides de forma aleatória. 
	# Para isso, use a função kmeans_init_centroids.
	#####################################################################

	# Inicializar K centroides aleatórios
	initial_centroids = kmeans_init_centroids(X, K)
	idx = find_closest_centroids(X, initial_centroids)

	# Compute initial means
	centroids = compute_centroids(X, idx, K)
	# Now run 10 iterations of K-means on fixed seeds
	max_iters = 10
	centroids, idx = run_kmeans(X, initial_centroids, max_iters, plot_progress=True)
	print('Centroids after the 1st update:\n' + str(centroids))
	
	# Plot final clustering.
	plt.scatter(X[np.where(idx==0),0],X[np.where(idx==0),1], marker='x')
	plt.scatter(X[np.where(idx==1),0],X[np.where(idx==1),1], marker='x')
	plt.scatter(X[np.where(idx==2),0],X[np.where(idx==2),1], marker='x')
	plt.title('Final clustering - Random Centroids')
	plt.show()
		

if __name__ == "__main__":
	main()
