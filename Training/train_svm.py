from Functions.load import *
from Functions.reshape_functions import *
from Functions.kfold import *
from Models.SVM import *
from Preprocessing.PCA import *
from Metrics.DCF import *
from Metrics.ROC import *

def SVM_diff_priors(D, L):
    C = 10
    priors = [
        (0.5, 0.5),
        (0.5, 0.1),
        (0.5, 0.9),
        (0.1, 0.5),
        (0.1, 0.1),
        (0.1, 0.9),
        (0.9, 0.5),
        (0.9, 0.1),
        (0.9, 0.9),
    ]

    for pi_T, pi in priors:
        svm = Linear_SVM(1, C)
        SPost, Label = kfold(D, L,svm, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")



def RadKernBased_diff_priors(D, L):
    C = 10
    lbd = 0.001
    priors = [
        (0.5, 0.5),
        (0.5, 0.1),
        (0.5, 0.9),
        (0.1, 0.5),
        (0.1, 0.1),
        (0.1, 0.9),
        (0.9, 0.5),
        (0.9, 0.1),
        (0.9, 0.9),
    ]

    for pi_T, pi in priors:
        svm = Radial_SVM(1, C, lbd)
        SPost, Label = kfold(D, L, svm, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")


def Poly_SVM_diff_priors(D, L):
    C = 0.001
    priors = [
        (0.5, 0.5),
        (0.5, 0.1),
        (0.5, 0.9),
        (0.1, 0.5),
        (0.1, 0.1),
        (0.1, 0.9),
        (0.9, 0.5),
        (0.9, 0.1),
        (0.9, 0.9),
    ]
    for pi_T, pi in priors:
        svm = Polynomial_SVM(1, 1, 2, C)
        SPost, Label = kfold(D, L, svm, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")
