from dp4gp import dp4gp
import random
import numpy as np
import GPy
import matplotlib.pyplot as plt
#from dp4gp import histogram
from dask_searchcv import GridSearchCV
#from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
import os
import distributed

"""
Usage recommendation:

getscores returns a matrix of all the k-folds and parameters.
getprobabilities returns a matrix, using the same k-folds and parameters of picking that combination

if you're just interested in the score for a given parameter combination then
call getscores with just one value of parameters, or maybe just vary epsilon?
"""


def install_libraries_on_workers(url,runlist = None):
    """Install libraries if necessary on workers etc.
    
    e.g. if already on server...
    install_libraries_on_workers('127.0.0.1:8786')
    """
    from dask.distributed import Client
    client = Client(url)
    
    if runlist is None:
        runlist = ['pip install -U pip','sudo apt install libgl1-mesa-glx -y','conda update scipy -y','pip install git+https://github.com/sods/paramz.git','pip install git+https://github.com/SheffieldML/GPy.git','pip install git+https://github.com/lionfish0/dp4gp.git','conda install dask-searchcv -c conda-forge -y', 'pip install git+https://github.com/lionfish0/dask_dp4gp.git', 'pip install numpy', 'conda remove argcomplete -y']#, 'conda install python=3.6 -y']

    for item in runlist:
        print("Installing '%s' on workers..." % item)
        client.run(os.system,item)
        print("Installing '%s' on scheduler..." % item)
        client.run_on_scheduler(os.system,item)    
        #os.system(item) #if you need to install it locally too
        
        

# This is an estimator for sklearn
class DPCloaking(BaseEstimator):
    def __init__(self, lengthscale=None, variance=None, errorlimit=None, kern=None, sensitivity=1.0, epsilon=1.0, delta=0.01, inducing=None, noisevariance=None, getxvalfoldsensitivities=False):
        """
        DPCloaking(lengthscale=None, variance=None, errorlimit=None, kern=None, sensitivity=1.0, epsilon=1.0, delta=0.01, inducing=None, noisevariance=None, getxvalfoldsensitivities=False)
        
        lengthscale=None - kernel lengthscale
        variance=None - kernel variance
        noisevariance=None - model gaussian white noise
        errorlimit=None - when using the class' score function to report the SSE
                we can set a limit on how large a single error is allowed to be.
        kern=None - a GPy kernel (needs to have lengthscales and variances)
        sensitivity=1.0 - the amount one output can change
        epsilon=1.0, delta=0.01 - DP parameters
        inducing = None - locations of inducing points, default to None - not using inducing points.
        getxvalfoldsensitivities=False - the score method will return the sum-squared-error
                of each fold by default, but if this is set to true it will return the
                sensitivity of the SSE caused by a perturbed point being in the training data. 
        """
        
        self.errorlimit = errorlimit
        self.kern = kern
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        self.inducing = inducing
        self.getxvalfoldsensitivities = getxvalfoldsensitivities

        if lengthscale is not None:
            self.kern.lengthscale = lengthscale
        if variance is not None:
            self.kern.variance = variance
        if noisevariance is not None:
            self.noisevariance = noisevariance
        else:
            self.noisevariance = 1.0
        
    def fit(self, X, y, **kwargs):
        """
        fit(X, y)
        
        Create the GPy model using the data in X and y, and the hyperparameters etc set by
        the constructor (or subsequently during a grid search)
        """
        self.kern.lengthscale = self.lengthscale
        self.kern.variance = self.variance
        if self.inducing is None:
            self.model = GPy.models.GPRegression(X,y,self.kern,normalizer=None)
            self.model.Gaussian_noise = self.noisevariance
            self.dpgp = dp4gp.DPGP_cloaking(self.model,self.sensitivity,self.epsilon,self.delta)
        else:
            if isinstance(self.inducing, list):
                inducinglocs = self.inducing
            else:
                inducinglocs = KMeans(n_clusters=self.inducing, random_state=0).fit(X).cluster_centers_
            self.model = GPy.models.SparseGPRegression(X,y,self.kern,normalizer=None,Z=inducinglocs)
            self.model.Gaussian_noise = self.noisevariance
            self.dpgp = dp4gp.DPGP_inducing_cloaking(self.model,self.sensitivity,self.epsilon,self.delta)
        return self

    def predict(self, X, Nattempts=2, Nits=5): #todo set Nits back to a larger value (e.g. 100)
        """
        predict(X,Nattempts=2, Nits=100)
        
        make predictions of the outputs, y, given inputs X.
        """
        ypred,_,_= self.dpgp.draw_prediction_samples(X,Nattempts=Nattempts,Nits=Nits)
        return ypred
    
    def score(self, X, y=None):
        """score(X,y=None)
        
        Return the (truncated) negative SSE or part of the sensitivity of the SSE
        
        If self.getxvalfoldsensitivities is True then, this isn't the actual score.
        Instead it returns the sensitivity of the sum squared error if the perturbed
        point lies in the training data.
        
        If it's false this returns the truncated NEGATIVE Sum Squared Error
        
            Note on truncation: we have also put a bound on the error of
            one data point - it must lie within errorlimit of the actual
            prediction. If it doesn't then the value is truncated.

            Note: This score is used to select the hyperparameter config
            used later in the cross-validation procedure. The truncation
            will therefore not cause an underestimate in the final score.
            The truncation allows us to put a bound on the sensitivity of
            the SSE. """        

        if self.getxvalfoldsensitivities:          
            C = self.dpgp.get_C(X) #NumTest x NumTrain
            #we sum over the square of this (times the sensitivity) to get,
            #             |d c_kj|_2^2
            #we then find the largest value of this to get,
            #             d^2 max_j |c_kj|_2^2
            #and this we return, giving the largest effect a training
            #point can have on the SSE.
            return np.max(np.sum((self.sensitivity * C)**2,0))
        else:          
            errors = (y-self.predict(X))
            errors[errors>self.errorlimit] = self.errorlimit
            errors[errors<-self.errorlimit] = -self.errorlimit
            return -np.sum((y-self.predict(X))**2)

def getprobabilities(X,y,p_grid, cv, ac_sens):
    """getprobabilities(X,y,p_grid, cv, ac_sens)
    
    - X and y: Inputs and outputs
    - p_grid: grid of parameters to search over
    - cv: Number of cross-validation folds (this is different from the outer number of x-val folds)
    
    Gets the probability of picking each of the options provided by p_grid given the data in X and y.
    The algorithm is as follows:
     - Find the sensitivity of the SSE for each parameter-values
     - Find the SSE of each parameter-values
     - Find the probability of selecting those parameter-values
    """
    kern = GPy.kern.RBF(2.0,lengthscale=25.0,variance=1.0)
    errorlimit = ac_sens*4.0
   
    ####find sensitivity of the SSE (for each param combo)
    #this call gets the sensivities not the scores:
    #TODO This probably should be done locally as it's quick.
    clf = GridSearchCV(estimator=DPCloaking(sensitivity=ac_sens, inducing=4, getxvalfoldsensitivities=True, kern=kern, errorlimit=errorlimit), param_grid=p_grid, cv = cv)
    clf.fit(X,y)

    nparamcombos = len(clf.cv_results_['mean_test_score'])
    temp_sens = np.zeros([clf.cv,nparamcombos])
    for k in range(clf.cv):
        temp_sens[k,:] = clf.cv_results_['split%d_test_score' % k]
    #sensitivity of the sum squared error:
    print(np.sort(temp_sens,axis=0))
    sse_sens = ac_sens**2 + 2*ac_sens*errorlimit + ac_sens**2*np.max(np.sum(np.sort(temp_sens,axis=0)[0:clf.cv-1,:],0))

    ####find the SSE (for each param combo)
    clf = GridSearchCV(estimator=DPCloaking(sensitivity=ac_sens, inducing=4, getxvalfoldsensitivities=False, kern=kern, errorlimit=errorlimit), param_grid=p_grid, cv = cv)
    clf.fit(X,y)

    nparamcombos = len(clf.cv_results_['mean_test_score'])
    temp_scores = np.zeros([clf.cv,nparamcombos])
    for k in range(clf.cv):
        temp_scores[k,:] = clf.cv_results_['split%d_test_score' % k]
    scores = np.sum(temp_scores,0)

    ####compute the probability of selecting that param combo using the exponential mechanism
    selection_epsilon = 1
    param_probabilities = np.exp(selection_epsilon*scores / (2*sse_sens))
    param_probabilities = param_probabilities / np.sum(param_probabilities)
    
    return param_probabilities

def getscores(X,y,p_grid,cv,ac_sens):
    """
    Compute the negative RMSE of each of the fold/param combos
    """
    kern = GPy.kern.RBF(2.0,lengthscale=25.0,variance=1.0)

    clf = GridSearchCV(estimator=DPCloaking(sensitivity=ac_sens, inducing=4, getxvalfoldsensitivities=False, kern=kern), scoring='neg_mean_squared_error', param_grid=p_grid, cv = cv)
    clf.fit(X,y)

    nparamcombos = len(clf.cv_results_['mean_test_score'])
    scores = np.zeros([clf.cv,nparamcombos])
    for k in range(clf.cv):
        scores[k,:] = clf.cv_results_['split%d_test_score' % k]
    return scores
    

