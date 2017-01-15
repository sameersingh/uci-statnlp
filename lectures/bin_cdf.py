import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

cdf = stats.binom.cdf
for n in [10, 25, 50]:
	x = np.linspace(0,n,100)
	plt.plot(x/n,cdf(x, n, 0.5), label='n='+str(n))
plt.xlabel("Proportion of Data points < nx")
plt.ylabel("Probability")
plt.legend(loc=2)
plt.savefig('bin_cdf.png')
plt.show()
