import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fish_data = pd.read_csv('../data/fish.csv', sep=";")

with pm.Model() as ZIP_reg:
    psi = pm.Beta('psi', 1, 1)

    alpha = pm.Normal('alpha', 0, 10)
    beta = pm.Normal('beta', 0, 10, shape=2)
    lam = pm.math.exp(
        alpha + beta[0] * fish_data['child'] + beta[1] * fish_data['camper'])

    y = pm.ZeroInflatedPoisson('y', lam, psi, observed=fish_data['count'])
    trace_ZIP_reg = pm.sample(2000)
chain_ZIP_reg = trace_ZIP_reg[100:]
pm.traceplot(chain_ZIP_reg)
plt.show()

children = [0, 1, 2, 3, 4]
fish_count_pred_0 = []
fish_count_pred_1 = []
thin = 5
for n in children:
    without_camper = chain_ZIP_reg['alpha'][::thin] + \
                     chain_ZIP_reg['beta'][:, 0][::thin] * n
    with_camper = without_camper + chain_ZIP_reg['beta'][:, 1][::thin]
    fish_count_pred_0.append(np.exp(without_camper))
    fish_count_pred_1.append(np.exp(with_camper))

plt.plot(children, fish_count_pred_0, 'bo', alpha=0.01)
plt.plot(children, fish_count_pred_1, 'ro', alpha=0.01)

plt.xticks(children)
plt.xlabel('Number of children', fontsize=14)
plt.ylabel('Fish caught', fontsize=14)
plt.plot([], 'bo', label='without camper')
plt.plot([], 'ro', label='with camper')
plt.legend(fontsize=14)
plt.show()
