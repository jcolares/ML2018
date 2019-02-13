import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#carrega os dados do arquivo
dados = pd.read_csv('ex2data2.txt', header = None ,names=['Test 1', 'Test 2', 'Resultado'])

#Separa os dados obtidos do arquivo em 2 conjuntos (dataframes), 
# de acordo com o conteúdo da coluna "Resultado".  
positivo = dados[dados['Resultado'].isin([1])]  
negativo = dados[dados['Resultado'].isin([0])]

#Plota o gráfico de dispersão conforme solicitado no item 4.1.
fig = plt.figure()
ax = fig.subplots()  
ax.axis([-1, 1.5, -1, 1.5])
ax.scatter(positivo['Test 1'], positivo['Test 2'], s=50, c='k', marker='+', label='y=1')  
ax.scatter(negativo['Test 1'], negativo['Test 2'], s=50, c='y', marker='o', label='y=0')  
ax.legend()  
ax.set_xlabel('Microchip Test 1')  
ax.set_ylabel('Microchip Test 2')  
fig.savefig('plot4.1.png')
plt.show()    
