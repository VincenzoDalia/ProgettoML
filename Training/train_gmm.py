import matplotlib.pyplot as plt
from Functions.load import *
from Functions.reshape_functions import *
from Functions.kfold import *
from Models.GMM import *
from Preprocessing.PCA import *
from Metrics.DCF import *
from Metrics.ROC import *
from Preprocessing.ZNorm import *



def GMM_diff_priors(D, L):
    for i in range(1, 3):
        for pi in [0.5, 0.1, 0.9]:
            gmm = GMM(i)
            SPost, Label = kfold( D, L, gmm, 5, None)
            res = MIN_DCF(pi, 1, 1, Label, SPost)
            print("GMM min_DCF pi = ", pi, str(2**i) + " components: ", round(res, 3))

    for pi in [0.5, 0.1, 0.9]:
        gmm = GMM_Tied(3)
        SPost, Label = kfold( D, L, gmm, 5, None)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(
            "Tied_GMM min_DCF pi = ", pi, str(2**3) + " components : ", round(res, 3)
        )

    D, _ = PCA(D, 11)

    for pi in [0.5, 0.1, 0.9]:
        gmm = GMM(2)
        SPost, Label = kfold( D, L, gmm, 5, None)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(
            "GMM min_DCF pi = ", pi, str(2**i) + " components + PCA(11): ", round(res, 3)
        ) 
