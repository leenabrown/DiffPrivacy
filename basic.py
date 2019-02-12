import numpy as np
import dp_stats as dps

### example of mean and variance
x = np.random.rand(10)
sum = 0
for i in range(10):
	sum = sum + x[i]

#print (sum / 10.0)

#print(x)
x_mu = dps.dp_mean( x, 1.0, 0.1 )
#print(x_mu)

hist = dps.dp_hist ( x, num_bins=3, epsilon=1.0, delta=0.1, histtype = 'discrete' )

print(hist)
