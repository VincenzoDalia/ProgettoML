import numpy as np
from Metrics.DCF import MIN_DCF
from Functions.kfold import kfold
from Preprocessing.PCA import PCA
from Preprocessing.ZNorm import znorm
from Models.LogisticRegression import BinaryLogisticRegression, QuadraticLogisticRegression
import matplotlib.pyplot as plt
from functools import partial

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
    plt.legend(loc='upper left')
    plt.savefig("Training/Logistic_Regression_Plot/" + name + ".pdf")
    plt.close()

def calculate_min_dcf_linear(D, L, prior, regression, l_values):
    value = [0.5, 0.1, 0.9]
    results = [(MIN_DCF(value[0], 1, 1, Label, SPost),
                MIN_DCF(value[1], 1, 1, Label, SPost),
                MIN_DCF(value[2], 1, 1, Label, SPost)) 
               for l in l_values 
               for SPost, Label in [kfold(D, L, regression(l), 5, prior)]]
    return zip(*results)

def simple_Logistic_Regression_Graph(D, L, prior):
    l_values = np.logspace(-5, 5, num=21)
    min_dcf_results_05, min_dcf_results_01, min_dcf_results_09 = calculate_min_dcf_linear(D, L, prior, BinaryLogisticRegression, l_values)
    plot_results(min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, "Logistic Regression Graph", "Logistic Regression Graph")

def ZNorm_Logistic_Regression_Graph(D, L, prior):
    D = znorm(D)
    l_values = np.logspace(-5, 5, num=21)
    min_dcf_results_05, min_dcf_results_01, min_dcf_results_09 = calculate_min_dcf_linear(D, L, prior, BinaryLogisticRegression, l_values)
    plot_results(min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, "Logistic Regression ZNorm", "Logistic Regression ZNorm Graph")

def PCA_Logistic_Regression_Graph(D, L, prior):
    D_pca = PCA(D, 11)
    l_values = np.logspace(-5, 5, num=21)
    min_dcf_results_05, min_dcf_results_01, min_dcf_results_09 = calculate_min_dcf_linear(D_pca, L, prior, BinaryLogisticRegression, l_values)
    plot_results(min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, "Logistic Regression PCA", "Logistic Regression PCA Graph")

def process_prior(D, L, l, prior):
    pi_T, pi = prior
    regression = BinaryLogisticRegression(l)
    SPost, Label = kfold(D, L, regression, 5, pi_T)
    res = MIN_DCF(pi, 1, 1, Label, SPost)
    print(f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")

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
    for prior in priors:
        process_prior(D, L, l, prior)

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
    for prior in priors:
        process_prior(D, L, l, prior)


def calculate_min_dcf_quadratic(l_values, D, L, prior, znorm=False):
    min_dcf_results = []
    if znorm:
        D = znorm(D)
    for i, l in enumerate(l_values):
        regression = QuadraticLogisticRegression(l)
        SPost, Label = kfold(regression, 5, D, L, prior)
        res = MIN_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_results.append(res)
        print(i)
    return min_dcf_results

def Quad_LR_RAW(D, L, prior):
    l_values = np.logspace(-5, 5, num=21)
    min_dcf_results_05 = calculate_min_dcf_quadratic(l_values, D, L, prior)
    min_dcf_results_05_znorm = calculate_min_dcf_quadratic(l_values, D, L, prior, znorm=True)
    
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
    
    
def Quad_LR_diff_priors(D, L, znorm=False):
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
    if znorm:
        D = znorm(D)
        
    regression = QuadraticLogisticRegression(l)
    
    for pi_T, pi in priors:
        SPost, Label = kfold(D, L, regression, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF{'_znorm' if znorm else ''} (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")