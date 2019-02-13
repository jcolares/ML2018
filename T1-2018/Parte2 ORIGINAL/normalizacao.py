import numpy as np

def normalizar_caracteristica(X, y):
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X_norm = (X - mean_X) / std_X
	
	###########################################
	# Seu c�digo aqui: complete com c�digo para 
	# normalizar y e definir as vari�veis 
	# y_norm, mean_y e std_y.
	###########################################
	
    return X_norm, y_norm, mean_X, std_X, mean_y, std_y

