import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import numpy as np

iris = sns.load_dataset('iris')

df = iris.query("species == ('setosa', 'versicolor')")
y_1 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
x_1 = df[x_n].values

with pm.Model() as model_1:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))

    mu = alpha + pm.math.dot(x_1, beta)
    theta = 1 / (1 + pm.math.exp(-mu))
    bd = pm.Deterministic('bd',
                          -alpha / beta[1] - beta[0] / beta[1] * x_1[:, 0])

    y1 = pm.Bernoulli('y1', p=theta, observed=y_1)

    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace_1 = pm.sample(5000, step=step, start=start)

chain_1 = trace_1[100:]
varnames = ['alpha', 'beta']
pm.traceplot(chain_1)
plt.show()

idx = np.argsort(x_1[:, 0])
bd = chain_1['bd'].mean(0)[idx]
plt.scatter(x_1[:, 0], x_1[:, 1], c=y_1)
plt.plot(x_1[:, 0][idx], bd, color='r')

bd_hpd = pm.hpd(chain_1['bd'])[idx]
plt.fill_between(x_1[:, 0][idx], bd_hpd[:, 0], bd_hpd[:, 1], color='r',
                 alpha=0.5)

plt.xlabel(x_n[0], fontsize=16)
plt.ylabel(x_n[1], fontsize=16)
plt.show()

corr = iris[iris['species'] != 'virginica'].corr()
mask = np.tri(*corr.shape).T
sns.heatmap(corr.abs(), mask=mask, annot=True)
plt.show()

# Unbalanced Classes
df = iris.query("species == ('setosa', 'versicolor')")
df = df[45:]
y_2 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
x_2 = df[x_n].values

with pm.Model() as model_2:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))

    mu = alpha + pm.math.dot(x_2, beta)
    theta = 1 / (1 + pm.math.exp(-mu))
    bd = pm.Deterministic('bd',
                          -alpha / beta[1] - beta[0] / beta[1] * x_2[:, 0])

    y2 = pm.Bernoulli('y1', p=theta, observed=y_2)

    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace_2 = pm.sample(5000, step=step, start=start)

chain_2 = trace_2[100:]
varnames = ['alpha', 'beta']
pm.traceplot(chain_2)
plt.show()

idx = np.argsort(x_2[:, 0])
bd = chain_2['bd'].mean(0)[idx]
plt.scatter(x_2[:, 0], x_2[:, 1], c=y_2)
plt.plot(x_2[:, 0][idx], bd, color='r')

bd_hpd = pm.hpd(chain_2['bd'])[idx]
plt.fill_between(x_2[:, 0][idx], bd_hpd[:, 0], bd_hpd[:, 1], color='r',
                 alpha=0.5)

plt.xlabel(x_n[0], fontsize=16)
plt.ylabel(x_n[1], fontsize=16)
plt.show()

