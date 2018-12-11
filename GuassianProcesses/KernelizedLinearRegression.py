import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1)
x = np.random.uniform(0, 10, size=20)
y = np.random.normal(np.sin(x), 0.2)
plt.plot(x, y, 'o')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.show()


def guass_kernel(x, n_knots=5, w=2):
    """
    Simple Gaussian radial kernel
    :param x:
    :param n_knots:
    :param w:
    :return:
    """
    knots = np.linspace(np.floor(x.min()), np.ceil(x.max()), n_knots)
    return np.array([np.exp(-(x - k) ** 2 / w) for k in knots])


n_knots = 5
with pm.Model() as kernel_model:
    gamma = pm.Cauchy('gamma', alpha=0, beta=1, shape=n_knots)
    sd = pm.Uniform('sd', 0, 10)
    mu = pm.math.dot(gamma, guass_kernel(x, n_knots))
    y1 = pm.Normal('y1', mu=mu, sd=sd, observed=y)
    kernel_trace = pm.sample(10000, step=pm.Metropolis())

chain = kernel_trace[1000:]
pm.traceplot(chain)
plt.show()

ppc = pm.sample_ppc(chain, model=kernel_model, samples=100)
plt.plot(x, ppc['y1'].T, 'ro', alpha=0.1)
plt.plot(x, y, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.show()

new_x = np.linspace(np.floor(x.min()), np.ceil(x.max()), 100)
k = guass_kernel(new_x, n_knots)
gamma_pred = chain['gamma']
for i in range(100):
    idx = np.random.randint(0, len(gamma_pred))
    y_pred = np.dot(gamma_pred[idx], k)
    plt.plot(new_x, y_pred, 'r-', alpha=0.1)
plt.plot(x, y, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.show()
