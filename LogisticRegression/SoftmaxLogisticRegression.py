import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import numpy as np
import theano.tensor as tt

iris = sns.load_dataset('iris')
y_s = pd.Categorical(iris['species']).codes
x_n = iris.columns[:-1]
x_s = iris[x_n].values

with pm.Model() as model_s:
    alpha = pm.Normal('alpha', mu=0, sd=2, shape=3)
    beta = pm.Normal('beta', mu=0, sd=2, shape=(4, 3))

    mu = alpha + pm.math.dot(x_s, beta)

    theta = tt.nnet.softmax(mu)

    y1 = pm.Categorical('y1', p=theta, observed=y_s)
    start = pm.find_MAP()
    step = pm.NUTS()
    trace_s = pm.sample(2000, step, start)

pm.traceplot(trace_s)
plt.show()

data_pred = trace_s['alpha'].mean(axis=0) + \
            np.dot(x_s, trace_s['beta'].mean(axis=0))

y_pred = []
for point in data_pred:
    y_pred.append(np.exp(point) / np.sum(np.exp(point), axis=0))

np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s)

with pm.Model() as model_sf:
    alpha = pm.Normal('alpha', mu=0, sd=2, shape=2)
    beta = pm.Normal('beta', mu=0, sd=2, shape=(4, 2))

    alpha_f = tt.concatenate([[0], alpha])
    beta_f = tt.concatenate([np.zeros((4, 1)), beta], axis=1)

    mu = alpha_f + pm.math.dot(x_s, beta_f)

    theta = tt.nnet.softmax(mu)

    y1 = pm.Categorical('y1', p=theta, observed=y_s)
    start = pm.find_MAP()
    step = pm.NUTS()
    trace_sf = pm.sample(2000, step, start)

pm.traceplot(trace_sf)
plt.show()
