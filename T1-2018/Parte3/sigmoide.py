import numpy as np

def sigmoide(z):
    return (1.00 / (1.00 + np.exp(-z)))
