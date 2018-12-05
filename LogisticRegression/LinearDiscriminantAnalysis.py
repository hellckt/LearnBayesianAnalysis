import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import numpy as np

iris = sns.load_dataset("iris")
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values

with pm.Model() as lda:
    mus = pm.Normal('mus', mu=0, sd=10, shape=2)
    sigmas = pm.Uniform('sigmas', lower=0, upper=10, shape=2)

    setosa = pm.Normal('setosa', mu=mus[0], sd=sigmas[0], observed=x_0[:50])
    versicolor = pm.Normal('versicolor', mu=mus[1], sd=sigmas[1],
                           observed=x_0[50:])

    bd = pm.Deterministic('bd', (mus[0] + mus[1]) / 2)

    start = pm.find_MAP()
    step = pm.NUTS()
    trace = pm.sample(5000, step=step, start=start)

pm.traceplot(trace)
plt.show()

plt.axvline(trace['bd'].mean(), ymax=1, color='r')
bd_hpd = pm.hpd(trace['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='r', alpha=0.5)

plt.plot(x_0, y_0, 'o', color='k')

plt.xlabel(x_n, fontsize=16)
plt.show()
