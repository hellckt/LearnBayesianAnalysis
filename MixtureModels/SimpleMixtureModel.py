import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

clusters = 3
n_cluster = [90, 50, 75]
n_total = sum(n_cluster)
means = [9, 21, 35]
std_devs = [2, 2, 2]

mix = np.random.normal(np.repeat(means, n_cluster),
                       np.repeat(std_devs, n_cluster))

sns.kdeplot(np.array(mix))
plt.xlabel('$x$', fontsize=14)
plt.show()
