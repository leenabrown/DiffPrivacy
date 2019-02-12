import numpy as np
import dp_stats as dps

### example of mean and variance
x = np.random.rand(10)
x_mu = dps.dp_mean( x, 1.0, 0.1 )
x_vr = dps.dp_var( x, 1.0, 0.1 )
print(x_mu)
print(x_vr)

### example of DP-PCA
d = 10    # data dimension
n = 100   # number of samples
k = 5     # true rank

### create covariance matrix
A = np.zeros((d,d))
for i in range(d):
    if i < k:
        A[i,i] = d - i
    else:
        A[i, i] = 1
mean = 0.0 * np.ones(d) # true mean of the samples

### generate n samples
samps = np.random.multivariate_normal(mean, A, n)    # [nxd]
sigma = np.dot(samps.T, samps) # sample covariance matrix

U,S,V = np.linalg.svd(sigma, full_matrices=True)
U_reduce = U[:,:k]
quality = np.trace(np.dot(np.dot(U_reduce.T,A),U_reduce))
print(quality)

sigma_dp = dps.dp_pca_sn(samps.T, epsilon = 0.1)
U_dp, S_dp, V_dp = np.linalg.svd(sigma_dp, full_matrices=True)
U_dp_reduce = U_dp[:,:k]
quality_dp = np.trace(np.dot(np.dot(U_dp_reduce.T,A),U_dp_reduce))
print(quality_dp)
