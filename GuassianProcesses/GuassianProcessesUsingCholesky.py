import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1)
eta = 1
rho = 0.5
sigma = 0.03

# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(x).flatten()


# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    D = np.sum(a ** 2, 1).reshape(-1, 1) + \
        np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return eta * np.exp(- rho * D)


N = 20  # number of training points.
n = 100  # number of test points.

# Sample some input points and noisy versions of the function evaluated at
# these points.
X = np.random.uniform(0, 10, size=(N, 1))
y = f(X) + sigma * np.random.randn(N)

K = kernel(X, X)
L = np.linalg.cholesky(K + sigma * np.eye(N))

# points we're going to make predictions at.
Xtest = np.linspace(0, 10, n).reshape(-1, 1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest)
sd_pred = (np.diag(K_) - np.sum(Lk ** 2, axis=0)) ** 0.5

plt.fill_between(Xtest.flat, mu - 2 * sd_pred, mu + 2 * sd_pred, color="r",
                 alpha=0.2)
plt.plot(Xtest, mu, 'r', lw=2)
plt.plot(X, y, 'o')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.show()
