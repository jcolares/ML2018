import pandas as pd
import numpy as np
import scipy.io
from scipy.spatial import distance

centroids = np.array([[3, 3], [6, 2], [8, 5]])
X = np.array([
[ 6.48212628 , 2.5508514 ],
[ 7.30278708 , 3.38015979],
[ 6.99198434 , 2.98706729],
[ 4.8255341  , 2.77961664],
[ 6.11768055 , 2.85475655],
[ 0.94048944 , 5.71556802]])

#K = np.size(centroids, 1)
K = len(centroids)
idx = np.zeros((len(X), 1), dtype=np.int8)
dist = np.zeros((len(X), 1))
for i in range(len(idx)):
    dist[i] = dist_nova = distance.euclidean(X[i],centroids[0])
    idx[i] = 0
    for j in range(K):
        dist_nova = distance.euclidean(X[i],centroids[j])
        if dist_nova < dist[i]:
            dist[i] = dist_nova 
            idx[i] = j
print(idx)
print(X)

'''
ind= idx[:]==1
ind = ind.T
print("ind: ", ind)
X2 = (X[ind])
print(X2)
'''
Xidx = np.concatenate( (idx ,X ), axis = 1)
print("Xidx :")
print(Xidx)
ind = Xidx[:,0]==1
print("ind: ",ind)
X1 = (X[ind])
print(X1)
print(np.mean(X1[:,0],axis=0), np.mean(X1[:,1],axis=0))
'''
print("here it goes X1")
X1 = (X[:,1]>4)
print(X1)
print(X[X1])

idxX = np.concatenate( (idx ,X ), axis = 1)
C[i] = idxX[:,0]==i
'''


