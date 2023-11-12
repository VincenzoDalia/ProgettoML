# In this file we will implement the MVG models (Simple, NaiveBayes, Tied, NaiveBayesTied)

import scipy
import numpy as np
from Utils.utils import *


### FUNCTIONS USED IN THE MVG MODELS ###

def calculate_covariance_and_mean(D):

    N_features = D.shape[1]
    print(f"N_features: {N_features}")

    # calculate the mean of each feature and center the data
    mu = mcol(D.mean(1))
    DC = D - mu

    # calculate the covariance matrix
    C = np.dot(DC, DC.T)/N_features

    return C, mu


def logpdf_GAU_ND(X, mu, C):

    logdetC = np.linalg.slogdet(C)[1]
    M = X.shape[0]
    Xc = X - mu
    const_term = -0.5*M*np.log(2*np.pi)
    second_term = -0.5*logdetC
    third_term = -0.5*np.dot(np.dot(Xc.T, np.linalg.inv(C)),
                             Xc).ravel()  # oppure .sum(0)

    logN = const_term + second_term + third_term

    return logN


class MVG:

    def __init__(self):
        self.name = "MVG"

    def train(self, DTR, LTR, DTE, LTE, prior):

        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.prior = prior

        S = []

        # calculate the covariance matrix and the mean for each class
        # and
        for i in range(self.LTR.max()+1):

            # get the data for the class i
            D_c = self.DTR[:, self.LTR == i]
            C, mu = calculate_covariance_and_mean(D_c)
            #This function calculate the log density and I save this results into a score matrix S
            log_dens = logpdf_GAU_ND(self.DTE, mu, C)
            S.append(mrow(log_dens))
        S = np.vstack(S)
            
        #Calculate the prior matrix starting from a ones-matrix. I multiply each element times the prior
        prior = numpy.ones(S.shape) * [[self.eff_prior], [1-self.eff_prior]]
        
        #Calculate the joint distribution
        logSJoint = S + numpy.log(prior)
        
        #Calculate the marginal sum
        logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
        
        #Calculate posterior probability
        logSPost = logSJoint - logSMarginal
            
        
