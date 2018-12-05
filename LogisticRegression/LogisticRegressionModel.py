import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import numpy as np

iris = sns.load_dataset('iris')

sns.stripplot(x="species", y="sepal_length", data=iris, jitter=True)
plt.show()

sns.pairplot(iris, hue="species", diag_kind='kde')
plt.show()

df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values

with pm.Model() as model_0:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    mu = alpha + pm.math.dot(x_0, beta)
    theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
    bd = pm.Deterministic('bd', -alpha / beta)

    yl = pm.Bernoulli('yl', theta, observed=y_0)

    start = pm.find_MAP()
    step = pm.NUTS()
    trace_0 = pm.sample(5000, step, start)

chain_0 = trace_0[100:]
varnames = ['alpha', 'beta', 'bd']
pm.traceplot(chain_0, varnames)
plt.show()

theta = chain_0['theta'].mean(axis=0)
idx = np.argsort(x_0)
plt.plot(x_0[idx], theta[idx], color='b', lw=3)
plt.axvline(chain_0['bd'].mean(), ymax=1, color='r')
bd_hpd = pm.hpd(chain_0['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='r', alpha=0.5)

plt.plot(x_0, y_0, 'o', color='k')
theta_hpd = pm.hpd(chain_0['theta'])[idx]
plt.fill_between(x_0[idx], theta_hpd[:, 0], theta_hpd[:, 1], color='b',
                 alpha=0.5)

plt.xlabel(x_n, fontsize=16)
plt.ylabel(r'$\theta$', rotation=0, fontsize=0)
plt.show()


def classsify(n, threshold):
    """
    A simple classifying function
    :param n:
    :param threshold:
    :return:
    """
    n = np.array(n)
    mu = chain_0['alpha'].mean() + chain_0['beta'].mean() * n
    prob = 1 / (1 + np.exp(-mu))
    return prob, prob > threshold


classsify([5, 5.5, 6], 0.4)
