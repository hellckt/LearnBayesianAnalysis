import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

ans = sns.load_dataset('anscombe')
x_3 = ans[ans.dataset == 'III']['x'].values
y_3 = ans[ans.dataset == 'III']['y'].values

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
beta_c, alpha_c = stats.linregress(x_3, y_3)[:2]
plt.plot(x_3, (alpha_c + beta_c * x_3), 'k',
         label='y = {:.2f} + {:.2f} * x'.format(alpha_c, beta_c))
plt.plot(x_3, y_3, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc=0, fontsize=14)
plt.subplot(1, 2, 2)
sns.kdeplot(y_3)
plt.xlabel('$y$', fontsize=16)
plt.show()
