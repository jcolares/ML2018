import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def mapFeature(X1, X2):
    X = np.column_stack((X1,X2))
    poly = PolynomialFeatures(degree = 6 )
    Z = poly.fit_transform(X)

    return(Z)

