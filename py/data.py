import numpy as np
def genXY(beta, T, mu, sigma, Mu, Sigma):
    '''
    Generate response and predictor data where:
      -) X_m ~ MVN(Mu[m], Sigma[m])
      -) Y_m ~ mu[m] + X_m * beta[m,:] + N(mu[m], sigma[m]^2)

    beta (array MxP):     regression coefficients to use
    T (list of integers): number of observations to generate for each M (typically the same)
    mu (array M):         intercept term for each model
    sigma (array M):      standard deviation of noise term
    Mu (array MxP):       mean of each predictor
    Sigma (array MxPxP):  covariance matrix for each set of predictors

    returns:
      -) X (list length M): list of predictor variables
      -) Y (list length M): list of response variables
    '''

    M,P = beta.shape

    X = [np.random.multivariate_normal(size=(T[m]), mean=Mu[m,:], cov=Sigma[m,:,:]) for
         m in range(M)]

    Y = [X[m].dot(beta[m,:]) + mu[m] + np.random.normal(size=T[m], loc=mu[m], scale=sigma[m]) for
         m in range(M)]

    return( X, Y )
    
    
