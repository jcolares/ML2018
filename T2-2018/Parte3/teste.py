import numpy as np 
from scipy import stats 
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
x = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
ax.plot(x, stats.norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
plt.show()