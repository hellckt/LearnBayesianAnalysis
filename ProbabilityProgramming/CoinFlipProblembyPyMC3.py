import numpy as np
import scipy.stats as stats
import pymc3 as pm
import matplotlib.pyplot as plt

np.random.seed(123)
n_experiments = 4
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size=n_experiments)

with pm.Model() as first_model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    y = pm.Bernoulli('y', p=theta, observed=data)

    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(1000, step=step, start=start)

    burnin = 100
    chain = trace[burnin:]
    pm.traceplot(chain, lines={'theta': theta_real})
    plt.show()

with first_model:
    step = pm.Metropolis()
    multi_trace = pm.sample(1000, step=step, njobs=4)

burin = 0
multi_chain = multi_trace[burnin:]
pm.traceplot(multi_chain, lines={'theta': theta_real})
plt.show()

pm.gelman_rubin(multi_chain)
pm.forestplot(multi_chain, varnames=['theta'])
plt.show()

pm.summary(multi_chain)

pm.autocorrplot(chain)

pm.effective_n(multi_chain)['theta']
