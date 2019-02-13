
import numpy as np
import matplotlib.pyplot as plt

'''
theta = np.matrix([[2, 3, 4]])
parametros = int(theta.ravel().shape[1])
#print(parametros, np.size(theta, 1))

#u = np.linspace(-1,1.5,50)
#v = np.linspace(-1,1.5,50)

u = np.arange(-1, 1.5, 0.1)
v = np.arange(-1, 1.5, 0.1)


a = np.zeros((len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        a[i][j]=u[i];v[j]

#a = np.array((u,v))

a = np.meshgrid(u,v,  indexing='xy')

print(np.shape(a))
'''

x = np.array([1,2,3,4,5,6])
#y = np.array([4,5,6,7,8,9])
y = x**3 + x**2 + 2*x + 1
#y = x.dot(2)

#A = np.array(np.concatenate((x, y), axis = 0))
#A = A.reshape(6,2)
A = np.array(np.concatenate((x, y), axis = 0)).reshape(6,2)
print(np.shape(A))
print(A)



fig = plt.figure()
ax = fig.subplots()
#ax.axis([-1, 1.5, -1, 1.5])
ax.plot(A)
#plt.savefig('teste.png')
plt.show()
