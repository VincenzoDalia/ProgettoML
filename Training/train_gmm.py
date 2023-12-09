import matplotlib.pyplot as plt
from Functions.load import *
from Functions.reshape_functions import *
from Functions.kfold import *
from Models.GMM import *
from Preprocessing.PCA import *
from Metrics.DCF import *
from Metrics.ROC import *
from Preprocessing.ZNorm import *



 
def run_gmm_model(gmm, components, pi, message, D, L):
    SPost, Label = kfold(D, L, gmm, 5, None)
    res = MIN_DCF(pi, 1, 1, Label, SPost)
    print(message, "pi = ", pi, str(2**components) + " components: ", round(res, 3))

def GMM_diff_priors(D, L):
    for i in range(1, 3):
        for pi in [0.5, 0.1, 0.9]:
            run_gmm_model(GMM(i), i, pi, "GMM min_DCF", D, L)

    for pi in [0.5, 0.1, 0.9]:
        run_gmm_model(GMM_Tied(3), 3, pi, "Tied_GMM min_DCF", D, L)
 
    D_pca, _ = PCA(D, 11)

    for pi in [0.5, 0.1, 0.9]:
        run_gmm_model(GMM(2), 2, pi, "GMM min_DCF + PCA(11)", D_pca, L)
        
    D_pca, _ = PCA(D, 10)

    for pi in [0.5, 0.1, 0.9]:
        run_gmm_model(GMM(2), 2, pi, "GMM min_DCF + PCA(10)", D_pca, L)
        
    D_pca, _ = PCA(D, 9)

    for pi in [0.5, 0.1, 0.9]:
        run_gmm_model(GMM(2), 2, pi, "GMM min_DCF + PCA(9)", D_pca, L)
        
        
def GMM_diff_priors_znorm(D, L):
    D = znorm(D)
    for i in range(1, 3):
        for pi in [0.5, 0.1, 0.9]:
            run_gmm_model(GMM(i), i, pi, "GMM min_DCF + ZNorm",D, L)

    for pi in [0.5, 0.1, 0.9]:
        run_gmm_model(GMM_Tied(3), 3, pi, "GMM Tied min_DCF + ZNorm", D, L)