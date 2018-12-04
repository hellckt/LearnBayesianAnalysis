import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

rates = [1, 2, 5]
scales = [1, 2, 3]

x = np.linspace(0, 20, 100)
f, ax = plt.subplots(len(rates), len(scales), sharex=True, sharey=True)
for i in range(len(rates)):
    for j in range(len(scales)):
        rate = rates[i]
        scale = scales[j]
        rv = stats.gamma(a=rate, scale=scale)
        ax[i, j].plot(x, rv.pdf(x))
        ax[i, j].plot(0, 0,
                      label="$\\alpha$ = {:3.2f}\n$\\theta$ = {:3.2f}".format(
                          rate, scale), alpha=0)
        ax[i, j].legend()
ax[2, 1].set_xlabel('$x$')
ax[1, 0].set_ylabel('$pdf(x)$')
