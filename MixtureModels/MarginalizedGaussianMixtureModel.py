import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

clusters = 3
n_cluster = [90, 50, 75]
n_total = sum(n_cluster)
means = [9, 21, 35]
std_devs = [2, 2, 2]

mix = np.random.normal(np.repeat(means, n_cluster),
                       np.repeat(std_devs, n_cluster))

with pm.Model() as model_mg:
    p = pm.Dirichlet('p', a=np.ones(clusters))

    means = pm.Normal('means', mu=[10, 20, 35], sd=2, shape=clusters)
    sd = pm.HalfCauchy('sd', 5, shape=clusters)

    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
    step = pm.Metropolis()
    trace_mg = pm.sample(2000, step=step)

chain_mg = trace_mg[500:]
varnames_mg = ['means', 'sd', 'p']
pm.traceplot(chain_mg, varnames_mg)
plt.show()


ppc = pm.sample_ppc(chain_mg, 50, model_mg)
for i in ppc['y']:
    sns.kdeplot(i, alpha=0.1, color='b')

sns.kdeplot(np.array(mix), lw=2, color='k')
plt.xlabel('$x$', fontsize=14)
plt.show()

