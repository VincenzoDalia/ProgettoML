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
    lambda_values = np.logspace(-5, 2, num=41)

    plt.figure()
    plt.xlabel("\u03BB")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title(title)

    plt.plot(lambda_values, min_dcf_05, label="minDCF(\u03C0 = 0.5)")
    plt.plot(lambda_values, min_dcf_01, label="minDCF(\u03C0 = 0.1)")
    plt.plot(lambda_values, min_dcf_09, label="minDCF(\u03C0 = 0.9)")
    plt.xlim(lambda_values[0], lambda_values[-1])
    plt.legend()

    plt.savefig("Training/LR/" + name + ".pdf")
    plt.close()
    
    """
def LR_RAW(D, L, prior):
    l_values = np.logspace(-5, 2, num=41)
    value = [0.5, 0.1, 0.9]
    name = "LR_RAW"

    min_dcf_results_05 = []
    min_dcf_results_01 = []
    min_dcf_results_09 = []

    regression = BinaryLogisticRegression(0)

    for i, l in enumerate(l_values):
        regression.l = l
        SPost, Label = kfold(D, L, regression, 5, prior)

        res = MIN_DCF(value[0], 1, 1, Label, SPost)
        min_dcf_results_05.append(res)

        res = MIN_DCF(value[1], 1, 1, Label, SPost)
        min_dcf_results_01.append(res)

        res = MIN_DCF(value[2], 1, 1, Label, SPost)
        min_dcf_results_09.append(res)

        print(i)

    plot_results(
        min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, name, "RAW"
    )
  """
  
def run_Logistic_Regression_Graph(D, L, prior):
    l_values = np.logspace(-5, 2, num=41)
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
        return f"min_DCF (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}"

    # Utilizzo di map per applicare la funzione ad ogni tupla di priors
    results = list(map(partial(process_prior, D=D, L=L), priors))

    # Stampa dei risultati
    for result in results:
        print(result)
