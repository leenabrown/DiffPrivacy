# dp-stats

dp-stats is a Python library for differentially private statistics and machine learning algorithms

## Contact

* Subscribe to our mailing list: [dp_stats@email.rutgers.edu](https://email.rutgers.edu/mailman/listinfo/dp_stats)

## Dependencies

dp-stats has the following dependencies:

* Python 3.5
* Numpy 1.10.4
* Scipy 0.17.0

## Downloading

You can download the repository at https://gitlab.com/dp-stats/dp-stats.git, or using

```
$ git clone https://gitlab.com/dp-stats/dp-stats.git
```

## Installation

To install:

```
$ cd /path/to/dp_stats
```

In the directory run either of the following:

```
$ python setup.py install
```

To use in your programs:

```
import dp_stats as dps
```

## Contributing

This package is in alpha, so bug reports, especially regarding implementation and parameter setting, are very welcome. If you would like to become a developer, feel free to contact the authors.

Requests for additional features/algorithms are also welcome, as are requests for tutorials.

## Testing

Please run the following code to see if the installation is correct.

```
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
```

## License

MIT license

## Acknowledgements

This development of this package was supported by support from the
following sources:

* National Science Foundation under award CCF-1453432
* National Institutes of Health under award 1R01DA040487-01A1
* Defense Advanced Research Projects Agency (DARPA) and Space and Naval Warfare Systems Center, Pacific (SSC Pacific) under contract No. N66001-15-C-4070. 

Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of DARPA or SSC Pacific.
