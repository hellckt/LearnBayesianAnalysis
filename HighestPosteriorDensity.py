# -*- encoding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from plot_post import plot_post


def naive_hpd(post):
    sns.kdeplot(post)
    HPD = np.percentile(post, [2.5, 97.5])
    plt.plot(HPD, [0, 0], label="HPD {:.2f} {:.2f}".format(*HPD), linewidth=4,
             color='k')
    plt.legend(fontsize=16)
    plt.xlabel(r"$\theta$", fontsize=14)
    plt.gca().axes.get_yaxis().set_ticks([])


# Unimodal distribution
# np.random.seed(1)
# post = stats.beta.rvs(5, 11, size=1000)
# naive_hpd(post)
# plt.xlim(0, 1)
# plt.show()

# Bimodal distribution
np.random.seed(1)
gauss_a = stats.norm.rvs(loc=4, scale=0.9, size=3000)
gauss_b = stats.norm.rvs(loc=-2, scale=1, size=2000)
mix_norm = np.concatenate((gauss_a, gauss_b))


# naive_hpd(mix_norm)
# plt.savefig('./images/01_08.png', dpi=300, figsize=(5.5, 5.5))

plot_post(mix_norm, roundto=2, alpha=0.05)
plt.legend(loc=0, fontsize=16)
plt.xlabel(r"$\theta$", fontsize=14)
plt.show()
