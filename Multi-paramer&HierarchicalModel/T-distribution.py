import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

x_values = np.linspace(-10, 10, 200)
for df in [1, 2, 5, 30, 100]:
    distri = stats.t(df)
    x_pdf = distri.pdf(x_values)
    plt.plot(x_values, x_pdf, label=r"$\nu$ = {}".format(df))

x_pdf = stats.norm.pdf(x_values)
plt.plot(x_values, x_pdf, label=r"$\nu = \infty$")
plt.xlabel('x')
plt.ylabel('p(x)', rotation=0)
plt.legend(loc=0, fontsize=14)
plt.xlim(-7, 7)
plt.show()
