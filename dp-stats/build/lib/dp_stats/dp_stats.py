def dp_mean( data_vect, epsilon=1.0, delta=0.1 ):
    """
    This function provides a differentially-private estimate of the mean of a vector.

    Input:

      data_vect = data bounded in the range [0,1]
      epsilon = privacy parameter, default 1.0
      delta = privacy parameter, default 0.1

    Output:

      a scalar.


    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> x = np.random.rand(10)
      >>> x_mu = dps.dp_mean( x, 1.0, 0.1 )
      [ 0.57438844]

    """

    import numpy as np

    if ( any(data_vect < 0.0) or any(data_vect > 1.0) ):
        print('ERROR: Input vector should have bounded entries in [0,1].')
        return
    elif epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    elif delta < 0.0 or delta > 1.0:
        print('ERROR: Delta should be bounded in [0,1].')
        return
    else:

        n = len(data_vect)
        f = np.mean(data_vect)
        if delta == 0:
            noise = np.random.laplace(loc = 0, scale = 1/float(n*epsilon), size = (1,1))
        else:
            sigma = (1.0/(n*epsilon))*np.sqrt(2*np.log(1.25/delta))
            noise = np.random.normal(0.0, sigma, 1)
        f += noise

        return f
        
def dp_var( data_vector,epsilon=1.0,delta=0.1 ):
    """
    This function provides a differentially-private estimate of the variance of a vector.

    Input:

      data_vector = data bounded in the range [0,1]
      epsilon = privacy parameter, default 1.0
      delta = privacy parameter, default 0.1

    Output:

      a scalar.

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> x = np.random.rand(10)
      >>> x_mu = dps.dp_var( x, 1.0, 0.1 )
      [ 0.37882534]

    """

    import numpy as np

    if any(data_vector < 0.0) or any(data_vector > 1.0):
        print('ERROR: Input vector should have bounded entries in [0,1].')
        return
    elif epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    elif delta < 0.0 or delta > 1.0:
        print('ERROR: Delta should be bounded in [0,1].')
        return
    else:
        n = len(data_vector);
        mu=np.mean(data_vector);
        var=(1.0/n)*sum((value - mu) ** 2 for value in data_vector);
        delf=3.0*(1.0-1.0/n)/n;

        if delta==0:
            noise = np.random.laplace(loc = 0, scale = delf/epsilon, size = (1,1))
        else:
            sigma = (3.0/(n*epsilon))*(1-1.0/n)*np.sqrt(2*np.log(1.25/delta));
            noise=np.random.normal(0.0, sigma, 1);

        var += noise;

        return var
        
def dp_hist ( data, num_bins=10, epsilon=1.0, delta=0.1, histtype = 'continuous' ):
    """
    This function provides a differentially-private estimate of the histogram of a vector.

    Input:

      data = data vector
      num_bins = number of bins for the histogram, default is 10
      epsilon = privacy parameter, default 1.0
      delta = privacy parameter, default 0.1
      histtype = a string indicating which type of histogram is desired ('continuous', or 'discrete'),
                 by default, histtype = 'continuous'

    Note that for discrete histogram, the user input "num_bins" is ignored.

    Output:

      dp_hist_counts = number of items in each bins
      bin_edges = location of bin edges

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> x = np.random.rand(10)
      >>> x_mu = dps.dp_hist( x, 5, 1.0, 0.1, 'continuous' )
      (array([ 0.81163273, -1.18836727, -1.18836727,  0.81163273, -0.18836727]), array([ 0.1832111 ,  0.33342489,  0.48363868,  0.63385247,  0.78406625,
    0.93428004]))

    """

    import numpy as np

    if epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    elif delta < 0.0 or delta > 1.0:
        print('ERROR: Delta should be bounded in [0,1].')
        return
    else:
        if histtype == 'discrete':
            num_bins = len( np.unique(data) )
        hist_counts = [0] * num_bins
        data_min = min(data)
        data_max = max(data)
        bin_edges = np.linspace(data_min, data_max, num_bins+1)
        interval = (data_max - data_min) + 0.000000000001
        
        for kk in data:
            loc = (kk - data_min) / interval
            index = int(loc * num_bins)
            hist_counts[index] += 1.0

        if delta==0:
            noise = np.random.laplace(loc = 0, scale = 1.0/epsilon, size = (1,len(hist_counts)))
        else:
            sigma = (1.0/epsilon)*np.sqrt(2*np.log(1.25/delta))
            noise = np.random.normal(0.0, sigma, len(hist_counts))

        hist_array=np.asarray(hist_counts)
        noise_array=np.asarray(noise)
        dp_hist_counts = hist_array+noise_array

        return ( dp_hist_counts, bin_edges )


def dp_pca_ag ( data, epsilon=1.0, delta=0.1 ):
    '''
    This function provides a differentially-private estimate using Analyze Gauss method
    of the second moment matrix of the data

    Input:

      data = data matrix, samples are in columns
      epsilon = privacy parameter, defaul
      hat_A = (\epsilon, \delta)-differentially-private estimate of A = data*data'

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> data = np.random.rand(10)
      >>> hat_A = dps.dp_pca_ag ( data, 1.0, 0.1 )
      [[ 1.54704321  2.58597112  1.05587101  0.97735922  0.03357301]
       [ 2.58597112  4.86708836  1.90975259  1.41030773  0.06620355]
       [ 1.05587101  1.90975259  1.45824498 -0.12231379 -0.83844168]
       [ 0.97735922  1.41030773 -0.12231379  1.47130207  0.91925544]
       [ 0.03357301  0.06620355 -0.83844168  0.91925544  1.06881321]]

    '''

    import numpy as np

    if any( np.diag( np.dot( data.transpose(), data ) ) ) > 1:
        print('ERROR: Each column in the data matrix should have 2-norm bounded in [0,1].')
        return
    elif epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    elif delta < 0.0 or delta > 1.0:
        print('ERROR: Delta should be bounded in [0,1].')
        return
    else:

        A = np.dot( data, data.transpose() )
        D = ( 1.0 / epsilon ) * np.sqrt( 2.0 * np.log( 1.25 / delta ) )
        m = len(A)
        temp = np.random.normal( 0, D, (m, m))
        temp2 = np.triu( temp )
        temp3 = temp2.transpose()
        temp4 = np.tril(temp3, -1)
        E = temp2 + temp4
        hat_A = A + E
        return hat_A


def dp_pca_sn ( data, epsilon=1.0 ):
    '''
    This function provides a differentially-private estimate using Symmetric Noise method
    of the second moment matrix of the data

    Input:

      data = data matrix, samples are in columns
      epsilon = privacy parameter, default 1.0

    Output:

      hat_A = (\epsilon, \delta)-differentially-private estimate of A = data*data'

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> data = np.random.rand(10)
      >>> hat_A = dps.dp_pca_sn ( data, 1.0 )
      [[ 1.54704321  2.58597112  1.05587101  0.97735922  0.03357301]
       [ 2.58597112  4.86708836  1.90975259  1.41030773  0.06620355]
       [ 1.05587101  1.90975259  1.45824498 -0.12231379 -0.83844168]
       [ 0.97735922  1.41030773 -0.12231379  1.47130207  0.91925544]
       [ 0.03357301  0.06620355 -0.83844168  0.91925544  1.06881321]]

    '''

    import numpy as np

    if any( np.diag( np.dot( data.transpose(), data ) ) ) > 1:
        print('ERROR: Each column in the data matrix should have 2-norm bounded in [0,1].')
        return
    elif epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    else:
        A = np.dot( data, data.transpose() )
        d = len(A)
        nsamples = d + 1
        sigma = ( 1.0 / ( 2.0 * epsilon ) )
        Z_mean = 0.0
        Z = np.random.normal(Z_mean, sigma, (d,nsamples))
        E = np.dot( Z, Z.transpose() )
        hat_A = A + E
        return hat_A


def dp_pca_ppm ( data, k, Xinit, epsilon=1.0,delta=0.1 ):
    '''
    This function provides a differentially-private estimate using Private Power method
    of the second moment matrix of the data

    Input:

      data = data matrix, samples are in columns
      k = reduced dimension
      Xinit = d x k size, initialization for the sampling
      epsilon = privacy parameter, default 1.0
      delta = privacy parameter, default 0.1

    Output:

      X = (\epsilon, \delta)-differentially-private estimate of the top-k subspace of A = data*data'

    Example:

      >>> import numpy as np
      >>> import dp_stats as dps
      >>> data = np.random.rand(10)
      >>> hat_A = dps.dp_pca_ppm ( data, 1.0, 0.1 )
      [[ 1.54704321  2.58597112  1.05587101  0.97735922  0.03357301]
       [ 2.58597112  4.86708836  1.90975259  1.41030773  0.06620355]
       [ 1.05587101  1.90975259  1.45824498 -0.12231379 -0.83844168]
       [ 0.97735922  1.41030773 -0.12231379  1.47130207  0.91925544]
       [ 0.03357301  0.06620355 -0.83844168  0.91925544  1.06881321]]

    '''

    import numpy as np

    if any( np.diag( np.dot( data.transpose(), data ) ) ) > 1:
        print('ERROR: Each column in the data matrix should have 2-norm bounded in [0,1].')
        return
    elif epsilon < 0.0:
        print('ERROR: Epsilon should be positive.')
        return
    elif delta < 0.0 or delta > 1.0:
        print('ERROR: Delta should be bounded in [0,1].')
        return
    else:
        A = np.dot( data, data.transpose() )
        d = np.size( A, 0 )
        U, S, V = np.linalg.svd( A )
        param = S[k-1] * np.log( d ) / ( S[k-1] - S[k] )
        L = round( 10 * param )

        sigma = ( 1.0 / epsilon ) * np.sqrt( 4.0 * k * L * np.log( 1.0 / delta ) )
        x_old = Xinit
        count = 0
        while count <= L:
            G_new = np.random.normal( 0, np.linalg.norm( x_old, np.inf ) * sigma, (d, k))
            Y = np.dot(A, x_old) + G_new
            count += 1
            q, r = np.linalg.qr(Y)
            x_old = q[:, 0:k ]
        X = x_old

        return X