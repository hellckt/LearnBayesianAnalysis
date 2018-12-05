import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

z = np.linspace(-10, 10, 100)

logistic = 1 / (1 + np.exp(-z))
plt.plot(z, logistic)
plt.xlabel('$z$', fontsize=18)
plt.ylabel('$logistic(z)$', fontsize=18)
