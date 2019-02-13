import numpy as np

def normalizarCaracteristica(X):     

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = ((X - X_mean) / X_std)
    
    return X_norm, X_mean, X_std

'''
def normalizar_caracteristica(X, y):  # Versão original 
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X_norm = (X - mean_X) / std_X
	
	###########################################
	# Seu código aqui: complete com código para 
	# normalizar y e definir as variáveis 
	# y_norm, mean_y e std_y.
	###########################################
    mean_y = np.mean(y, axis=0)
    std_y = np.std(y, axis=0)
    y_norm = (y - mean_y) / std_y
	
    return X_norm, y_norm, mean_X, std_X, mean_y, std_y
'''
