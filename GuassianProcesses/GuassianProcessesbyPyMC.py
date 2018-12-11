import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import seaborn as sns

N = 20  # number of training points.
n = 100  # number of test points.

f = lambda x: np.sin(x).flatten()

x = np.random.uniform(0, 10, size=N)
y = f(x) + 0.03 * np.random.randn(N)


def squared_distance(x, y):
    return np.array(
        [[(x[i] - y[j]) ** 2 for i in range(len(x))] for j in range(len(y))])


with pm.Model() as GP:
    mu = np.zeros(N)
    eta = pm.HalfCauchy('eta', 5)
    rho = pm.HalfCauchy('rho', 5)
    sigma = pm.HalfCauchy('sigma', 5)

    D = squared_distance(x, x)

    K = tt.fill_diagonal(eta * pm.math.exp(-rho * D), eta + sigma)

    obs = pm.MvNormal('obs', mu, cov=K, observed=y)

    test_points = np.linspace(0, 10, 100)
    D_pred = squared_distance(test_points, test_points)
    D_off_diag = squared_distance(x, test_points)

    K_oo = eta * pm.math.exp(-rho * D_pred)
    K_o = eta * pm.math.exp(-rho * D_off_diag)

    mu_post = pm.Deterministic('mu_post', pm.math.dot(
        pm.math.dot(K_o, tt.nlinalg.matrix_inverse(K)), y))
    SIGMA_post = pm.Deterministic('SIGMA_post', K_oo - pm.math.dot(
        pm.math.dot(K_o, tt.nlinalg.matrix_inverse(K)), K_o.T))

    start = pm.find_MAP()
    trace = pm.sample(1000, start=start)

varnames = ['eta', 'rho', 'sigma']
chain = trace[100:]
pm.traceplot(chain, varnames)
plt.show()
