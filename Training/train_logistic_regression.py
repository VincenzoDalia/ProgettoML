import numpy as np
from Metrics.DCF import *
from Functions.kfold import *
from Preprocessing.PCA import *
from Preprocessing.ZNorm import *
from Models.LogisticRegression import *
import matplotlib.pyplot as plt
from functools import partial

##### DA MODIFICARE #####
def plot_results(min_dcf_05, min_dcf_01, min_dcf_09, name, title):
    
    lambda_values = np.logspace(-5, 5, num=21)

    plt.figure()
    plt.xlabel("Lambda")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title(title)

    plt.plot(lambda_values, min_dcf_05, label="minDCF(prior = 0.5)", linestyle='-', color='black')
    plt.plot(lambda_values, min_dcf_01, label="minDCF(prior = 0.1)", linestyle='--', color='green')
    plt.plot(lambda_values, min_dcf_09, label="minDCF(prior = 0.9)", linestyle='-.', color='red')
    
    plt.scatter(lambda_values[np.argmin(min_dcf_05)], min_dcf_05[np.argmin(min_dcf_05)], color='black')
    plt.scatter(lambda_values[np.argmin(min_dcf_01)], min_dcf_01[np.argmin(min_dcf_01)], color='green')
    plt.scatter(lambda_values[np.argmin(min_dcf_09)], min_dcf_09[np.argmin(min_dcf_09)], color='red')
    
    plt.xlim(lambda_values[0], lambda_values[-1])
    # Sposta la legenda in una posizione migliore
    plt.legend(loc='upper left')

    plt.savefig("Training/Logistic_Regression_Plot/" + name + ".pdf")
    plt.close()


        
def simple_Logistic_Regression_Graph(D, L, prior):
    l_values = np.logspace(-5, 5, num=21)
    value = [0.5, 0.1, 0.9]
    name = "Logistic Regression Graph"

    regression = BinaryLogisticRegression(0)
    
    def calculate_min_dcf_wrapper(l):
        print(l)
        regression.l = l
        SPost, Label = kfold(D, L, regression, 5, prior)
        return (
            MIN_DCF(value[0], 1, 1, Label, SPost),
            MIN_DCF(value[1], 1, 1, Label, SPost),
            MIN_DCF(value[2], 1, 1, Label, SPost)
        )  
        

    results = map(calculate_min_dcf_wrapper, l_values)

    min_dcf_results_05, min_dcf_results_01, min_dcf_results_09 = zip(*results)

    plot_results(
        min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, name, "Logistic Regression Graph"
    )

def ZNorm_Logistic_Regression_Graph(D, L, prior):
    D = znorm(D)
    name = "Logistic Regression ZNorm"
    l_values = np.logspace(-5, 5, num=21)

    value = [0.5, 0.1, 0.9]

    min_dcf_results_05 = []
    min_dcf_results_01 = []
    min_dcf_results_09 = []

    regression =  regression = BinaryLogisticRegression(0)

    def calculate_min_dcf_wrapper(l):
        print(l)
        regression.l = l
        SPost, Label = kfold(D, L, regression, 5, prior)
        return (
            MIN_DCF(value[0], 1, 1, Label, SPost),
            MIN_DCF(value[1], 1, 1, Label, SPost),
            MIN_DCF(value[2], 1, 1, Label, SPost)
        )

    results = map(calculate_min_dcf_wrapper, l_values)

    min_dcf_results_05, min_dcf_results_01, min_dcf_results_09 = zip(*results)

    plot_results(
        min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, name, "Logistic Regression ZNorm Graph"
    )

def PCA_Logistic_Regression_Graph(D, L, prior):
    name = "Logistic Regression PCA"
    l_values = np.logspace(-5, 5, num=21)
    value = [0.5, 0.1, 0.9]
    regression = BinaryLogisticRegression(0)

    def calculate_min_dcf_wrapper(l):
        print(l)
        regression.l = l
        SPost, Label = kfold(D, L, regression, 5, prior)
        return (
            MIN_DCF(value[0], 1, 1, Label, SPost),
            MIN_DCF(value[1], 1, 1, Label, SPost),
            MIN_DCF(value[2], 1, 1, Label, SPost)
        )

    min_dcf_results_05, min_dcf_results_01, min_dcf_results_09 = zip(*[calculate_min_dcf_wrapper(l) for l in l_values])

    plot_results(
        min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, name, "Logistic Regression PCA Graph"
    )

def Quad_LR_RAW(D, L, prior):
    l_values = np.logspace(-5, 5, num=20)

    value = [0.5]

    min_dcf_results_05 = []
    min_dcf_results_05_znorm = []

    for i, l in enumerate(l_values):
        regression = QuadraticLogisticRegression(l)

        SPost_1, Label_1 = kfold(regression, 5, D, L, prior)
        res_1 = MIN_DCF(value[0], 1, 1, Label_1, SPost_1)
        min_dcf_results_05.append(res_1)
        print(i)

    D = znorm(D)
    for i, l in enumerate(l_values):
        regression = QuadraticLogisticRegression(l)

        SPost_2, Label_2 = kfold(regression, 5, D, L, prior)
        res_2 = MIN_DCF(value[0], 1, 1, Label_2, SPost_2)
        min_dcf_results_05_znorm.append(res_2)
        print(i)

    plt.figure()
    plt.xlabel("\u03BB")
    plt.xscale("log")
    plt.ylabel("minDCF")

    plt.plot(l_values, min_dcf_results_05, label="minDCF(\u03C0 = 0.5) RAW")
    plt.plot(l_values, min_dcf_results_05_znorm, label="minDCF(\u03C0 = 0.5) Z-norm")

    plt.xlim(l_values[0], l_values[-1])
    plt.legend()
    plt.savefig("Training/LR/Plot/Quad_LR.pdf")
    plt.close()
     
def LR_diff_priors(D, L):
    l = 0.01
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
    
    # Definizione della funzione interna per elaborare le tuple di priors
    def process_prior(prior, D, L):
        pi_T, pi = prior
        regression = BinaryLogisticRegression(l)
        SPost, Label = kfold(D, L, regression, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")
        #return f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}"

    # Utilizzo di map per applicare la funzione ad ogni tupla di priors
    list(map(partial(process_prior, D=D, L=L), priors))
 
def LR_diff_priors_zscore(D, L):
    l = 0.0001
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
    D = znorm(D)
    for pi_T, pi in priors:
        regression = BinaryLogisticRegression(l)
        SPost, Label = kfold(D, L, regression, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF znorm (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}") 
    
def Quad_LR_diff_priors(D, L):
    l = 100
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
        regression = QuadraticLogisticRegression(l)
        SPost, Label = kfold(D, L, regression, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")
