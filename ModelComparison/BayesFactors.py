import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns

coins = 300
heads = 90
y = np.repeat([0, 1], [coins - heads, heads])

with pm.Model() as model_BF:
    p = np.array([0.5, 0.5])
    model_index = pm.Categorical('model_index', p=p)
    m_0 = (4, 8)
    m_1 = (8, 4)
    m = pm.math.switch(pm.math.eq(model_index, 0), m_0, m_1)

    theta = pm.Beta('theta', m[0], m[1])
    y = pm.Bernoulli('y', theta, observed=y)
    trace_BF = pm.sample(5000)

chain_BF = trace_BF[500:]
pm.traceplot(chain_BF)
plt.show()

pM1 = chain_BF['model_index'].mean()
pM0 = 1 - pM1
BF = (pM0 / pM1) * (p[1] / p[0])

with pm.Model() as model_BF_0:
    theta = pm.Beta('theta', 4, 8)
    y = pm.Bernoulli('y', theta, observed=y)

    trace_BF_0 = pm.sample(5000)
chain_BF_0 = trace_BF_0[500:]
pm.traceplot(trace_BF_0)
plt.show()

with pm.Model() as model_BF_1:
    theta = pm.Beta('theta', 8, 4)
    y = pm.Bernoulli('y', theta, observed=y)

    trace_BF_1 = pm.sample(5000)
chain_BF_1 = trace_BF_1[500:]
pm.traceplot(chain_BF_1)
plt.show()

# the smaller the better
waic_0 = pm.waic(chain_BF_0, model_BF_0)
waic_1 = pm.waic(chain_BF_1, model_BF_1)

loo_0 = pm.loo(chain_BF_0, model_BF_0)
loo_1 = pm.loo(chain_BF_1, model_BF_1)

plt.figure(figsize=(8, 4))
plt.subplot(121)
for idx, ic in enumerate((waic_0, waic_1)):
    plt.errorbar(ic[0], idx, xerr=ic[1], fmt='bo')
plt.title('WAIC')
plt.yticks([0, 1], ['model_0', 'model_1'])
plt.ylim(-1, 2)

plt.subplot(122)
for idx, ic in enumerate((loo_0, loo_1)):
    plt.errorbar(ic[0], idx, xerr=ic[1], fmt='go')
plt.title('LOO')
plt.yticks([0, 1], ['model_0', 'model_1'])
plt.ylim(-1, 2)

plt.tight_layout()
plt.show()
