import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
x_values = np.linspace(-10, 10, 300)
for df in [1, 2, 5, 15]:
    distri = stats.laplace(scale=df)
    x_pdf = distri.pdf(x_values)
    plt.plot(x_values, x_pdf, label="$b$ = {}".format(df))

x_pdf = stats.norm.pdf(x_values)
plt.plot(x_values, x_pdf, label="Gaussian")
plt.xlabel('x')
plt.ylabel('p(x)', rotation=0)
plt.legend(loc=0, fontsize=14)
plt.xlim(-7, 7);
plt.show()
