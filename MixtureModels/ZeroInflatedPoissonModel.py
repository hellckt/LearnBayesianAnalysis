import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
n = 100
theta = 2.5  # Poisson rate
pi = 0.1  # probability of extra-zeros (pi = 1-psi)

# Simulate some data
counts = np.array(
    [(np.random.random() > pi) * np.random.poisson(theta) for i in range(n)])

with pm.Model() as ZIP:
    psi = pm.Beta('p', 1, 1)
    lam = pm.Gamma('lam', 2, 0.1)

    y = pm.ZeroInflatedPoisson('y', lam, psi, observed=counts)
    trace = pm.sample(5000)
pm.traceplot(trace[100:])
plt.show()
