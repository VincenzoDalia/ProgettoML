import numpy as np
from Metrics.DCF import *
from Functions.kfold import kfold
from Preprocessing.PCA import PCA
from Preprocessing.ZNorm import znorm
from Models.LogisticRegression import BinaryLogisticRegression, QuadraticLogisticRegression
import matplotlib.pyplot as plt
from functools import partial
from Calibration.Calibrate import *
from Metrics.BayesErr import *

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

# Comparation plot with raw, znorm, pca 11 and pca 11 + znorm
def comparation_plot(D, L, prior):
    l_values = np.logspace(-5, 5, num=21)
    regression = BinaryLogisticRegression
    
    min_dcf_results_raw = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D, L, regression(l), 5, prior)]]    
    
    D_Znorm = znorm(D)
    min_dcf_results_znorm = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_Znorm, L, regression(l), 5, prior)]]
    
    print("RAW done")
    
    D_PCA_11, _ = PCA(D, 11)
    min_dcf_results_pca_11 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_11, L, regression(l), 5, prior)]]
    
    D_PCA_Znorm_11, _  = PCA(D_Znorm, 11)
    min_dcf_results_pca_znorm_11 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_Znorm_11, L, regression(l), 5, prior)]]
    
    print("PCA 11 done")
    
    
    
    plt.figure()
    plt.xlabel("Lambda")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Linear Logistic Regression Comparison")
    
    plt.plot(l_values, min_dcf_results_raw, label="minDCF(RAW)", color='blue')
    plt.plot(l_values, min_dcf_results_znorm, label="minDCF(Z-norm)",  color='green')
    plt.plot(l_values, min_dcf_results_pca_11, label="minDCF(PCA 11)",  color='orange')
    plt.plot(l_values, min_dcf_results_pca_znorm_11, label="minDCF(PCA 11 + Z-norm)",  color='red')
    
    plt.xlim(l_values[0], l_values[-1])
    plt.legend(loc='upper left')
    plt.savefig("Training/Logistic_Regression_Plot/Linear Logistic Regression Comparison.pdf")
    plt.close()
    
# Comparation plot 2 with PCA 10 and 9 (with and without znorm)
def comparation_plot_2(D, L, prior):
    
    l_values = np.logspace(-5, 5, num=21)
    regression = BinaryLogisticRegression
    
    D_Znorm = znorm(D)
    
    D_PCA_10, _ = PCA(D, 10)
    min_dcf_results_pca_10 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_10, L, regression(l), 5, prior)]]
    
    D_PCA_Znorm_10, _ = PCA(D_Znorm, 10)
    min_dcf_results_pca_znorm_10 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_Znorm_10, L, regression(l), 5, prior)]]
    
    print("PCA 10 done")
    
    D_PCA_9, _ = PCA(D, 9)
    min_dcf_results_pca_9 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_9, L, regression(l), 5, prior)]]
    
    D_PCA_Znorm_9, _  = PCA(D_Znorm, 9)
    min_dcf_results_pca_znorm_9 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_Znorm_9, L, regression(l), 5, prior)]]
    
    print("PCA 9 done")
    
    plt.figure()
    plt.xlabel("Lambda")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Linear Logistic Regression Comparison 2")
    
    plt.plot(l_values, min_dcf_results_pca_10, label="minDCF(PCA 10)",  color='blue')
    plt.plot(l_values, min_dcf_results_pca_znorm_10, label="minDCF(PCA 10 + Z-norm)",  color='green')
    plt.plot(l_values, min_dcf_results_pca_9, label="minDCF(PCA 9)", color='orange')
    plt.plot(l_values, min_dcf_results_pca_znorm_9, label="minDCF(PCA 9 + Z-norm)",  color='red')
    
    plt.xlim(l_values[0], l_values[-1])
    plt.legend(loc='upper left')
    plt.savefig("Training/Logistic_Regression_Plot/Linear Logistic Regression Comparison 2.pdf")
    plt.close()
 
def simple_Logistic_Regression_Graph(D, L, prior):
    l_values = np.logspace(-5, 5, num=21)
    min_dcf_results_05, min_dcf_results_01, min_dcf_results_09 = calculate_min_dcf_linear(D, L, prior, BinaryLogisticRegression, l_values)
    plot_results(min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, "Logistic Regression Graph", "Logistic Regression Graph")

def ZNorm_Logistic_Regression_Graph(D, L, prior):
    D = znorm(D)
    l_values = np.logspace(-5, 5, num=21)
    min_dcf_results_05, min_dcf_results_01, min_dcf_results_09 = calculate_min_dcf_linear(D, L, prior, BinaryLogisticRegression, l_values)
    plot_results(min_dcf_results_05, min_dcf_results_01, min_dcf_results_09, "Logistic Regression ZNorm", "Logistic Regression ZNorm Graph")

def different_Pi_T_Raw_Graph(D, L):
    l_values = np.logspace(-5, 5, num=21)
    regression = BinaryLogisticRegression
    
    results_raw_Pi_T_05 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D, L, regression(l), 5, 0.5)]]
    
    results_raw_Pi_T_01 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D, L, regression(l), 5, 0.1)]]
    
    results_raw_Pi_T_09 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D, L, regression(l), 5, 0.9)]]
    
    plt.figure()
    plt.xlabel("Lambda")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Linear Logistic Regression RAW")

    plt.plot(l_values, results_raw_Pi_T_05, label="minDCF(Pi_T = 0.5)", color='blue')
    plt.plot(l_values, results_raw_Pi_T_01, label="minDCF(Pi_T = 0.1)", color='green')
    plt.plot(l_values, results_raw_Pi_T_09, label="minDCF(Pi_T = 0.9)", color='red')
    
    plt.xlim(l_values[0], l_values[-1])
    plt.legend()
    plt.savefig("Training/Logistic_Regression_Plot/Linear Logistic Regression RAW - Different Pi_t.pdf")
    plt.close()
     
def different_Pi_T_znorm_Graph(D, L):
    
    l_values = np.logspace(-5, 5, num=21)
    regression = BinaryLogisticRegression
    D_Znorm = znorm(D)
    
    results_znorm_Pi_T_05 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_Znorm, L, regression(l), 5, 0.5)]]
    
    results_znorm_Pi_T_01= [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_Znorm, L, regression(l), 5, 0.1)]]
    
    results_znorm_Pi_T_09 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_Znorm, L, regression(l), 5, 0.9)]]
    
    plt.figure()
    plt.xlabel("Lambda")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Linear Logistic Regression ZNorm ")

    plt.plot(l_values, results_znorm_Pi_T_05, label="minDCF(Pi_T = 0.5)", color='blue')
    plt.plot(l_values, results_znorm_Pi_T_01, label="minDCF(Pi_T = 0.1)", color='green')
    plt.plot(l_values, results_znorm_Pi_T_09, label="minDCF(Pi_T = 0.9)", color='red')
    
    plt.xlim(l_values[0], l_values[-1])
    plt.legend()
    plt.savefig("Training/Logistic_Regression_Plot/Linear Logistic Regression ZNorm - Different Pi_T.pdf")
    plt.close()
        
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

def PCA_Logistic_Regression_Graph_various_Pi(D, L, pi):
    l_values = np.logspace(-5, 5, num=21)
    regression = BinaryLogisticRegression
    
    results_raw = [MIN_DCF(pi, 1, 1, Label, SPost)
                     for l in l_values 
                     for SPost, Label in [kfold(D, L, regression(l), 5, 0.5)]]
    
    D_PCA_11, _ = PCA(D, 11)
    
    results_PCA_11 = [MIN_DCF(pi, 1, 1, Label, SPost)
                        for l in l_values 
                        for SPost, Label in [kfold(D_PCA_11, L, regression(l), 5, 0.5)]]
    
    D_PCA_10, _ = PCA(D, 10)
    
    results_PCA_10 = [MIN_DCF(pi, 1, 1, Label, SPost)
                        for l in l_values 
                        for SPost, Label in [kfold(D_PCA_10, L, regression(l), 5, 0.5)]]
    
    D_PCA_9, _ = PCA(D, 9)
    
    results_PCA_9 = [MIN_DCF(pi, 1, 1, Label, SPost)
                        for l in l_values 
                        for SPost, Label in [kfold(D_PCA_9, L, regression(l), 5, 0.5)]]
    
    plt.figure()
    plt.xlabel("Lambda")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title(f"Linear Logistic Regression PCA Pi = {pi}  Pi_T = 0.5")
    
    plt.plot(l_values, results_raw, label="minDCF(RAW)", color='blue')
    plt.plot(l_values, results_PCA_11, label="minDCF(PCA 11)", color='green')
    plt.plot(l_values, results_PCA_10, label="minDCF(PCA 10)", color='orange')
    plt.plot(l_values, results_PCA_9, label="minDCF(PCA 9)", color='red')
    
    plt.xlim(l_values[0], l_values[-1])
    plt.legend()
    plt.savefig(f"Training/Logistic_Regression_Plot/Linear Logistic Regression PCA different pi- Pi = {pi} - Pi_T = 0.5.pdf")
    
def PCA_Logistic_Regression_Graph_various_Pi_T(D, L, prior):
    l_values = np.logspace(-5, 5, num=21)
    regression = BinaryLogisticRegression
    
    results_raw = [MIN_DCF(0.5, 1, 1, Label, SPost)
                     for l in l_values 
                     for SPost, Label in [kfold(D, L, regression(l), 5, prior)]]
    
    D_PCA_11, _ = PCA(D, 11)
    
    results_PCA_11 = [MIN_DCF(0.5, 1, 1, Label, SPost) 
                        for l in l_values 
                        for SPost, Label in [kfold(D_PCA_11, L, regression(l), 5, prior)]]
    
    D_PCA_10, _ = PCA(D, 10)
    
    results_PCA_10 = [MIN_DCF(0.5, 1, 1, Label, SPost)
                        for l in l_values 
                        for SPost, Label in [kfold(D_PCA_10, L, regression(l), 5, prior)]]
    
    D_PCA_9, _ = PCA(D, 9)
    
    results_PCA_9 = [MIN_DCF(0.5, 1, 1, Label, SPost)
                        for l in l_values 
                        for SPost, Label in [kfold(D_PCA_9, L, regression(l), 5, prior)]]   
    
    plt.figure()
    plt.xlabel("Lambda")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title(f"Linear Logistic Regression PCA - Pi = 0.5 Pi_T = {prior}")
    
    plt.plot(l_values, results_raw, label="minDCF(RAW)", color='blue')
    plt.plot(l_values, results_PCA_11, label="minDCF(PCA 11)", color='green')
    plt.plot(l_values, results_PCA_10, label="minDCF(PCA 10)", color='orange')
    plt.plot(l_values, results_PCA_9, label="minDCF(PCA 9)", color='red')
    
    plt.xlim(l_values[0], l_values[-1])
    plt.legend()
    plt.savefig(f"Training/Logistic_Regression_Plot/Linear Logistic Regression PCA different pi_t  - Pi = 0.5 - Pi_T = {prior}.pdf")
    plt.close()
  
  
def LR_candidate(D,L):
    
    l = 0.01
    pi_T = 0.1
    
    lr = BinaryLogisticRegression(l)
    
    SPost,Label = kfold(D,L,lr,5,pi_T)
    
    return SPost,Label


def calibrated_LR_dcf(D, L, prior):
    print(f"LR - min_dcf / act_dcf  {prior} \n")
    llr, Label = LR_candidate(D, L)
    llr_cal, Label_cal = calibrate(llr, Label, 0.5)
    predicted_labels = optimal_bayes_decision(llr_cal, prior, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = MIN_DCF(prior, 1, 1, Label_cal, llr_cal)
    act_dcf = DCF(prior, conf_matrix, 1, 1, True)
    print("LR (train) min_dcf: ", round(min_dcf, 3))
    print("LR (train) act_dcf: ", round(act_dcf, 3))
  
### ---------------------- Quadratic Logistic Regression ---------------------- ###

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
    

# Quadratic comparation plot with raw, znorm, pca 11 and pca 11 + znorm
def quadratic_comparation_plot(D, L, prior):
    l_values = np.logspace(-5, 5, num=21)
    regression = QuadraticLogisticRegression
    
    min_dcf_results_raw = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D, L, regression(l), 5, prior)]]    
    
    D_Znorm = znorm(D)
    min_dcf_results_znorm = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_Znorm, L, regression(l), 5, prior)]]
    
    print("RAW done")
    
    D_PCA_11, _ = PCA(D, 11)
    min_dcf_results_pca_11 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_11, L, regression(l), 5, prior)]]
    
    D_PCA_Znorm_11, _  = PCA(D_Znorm, 11)
    min_dcf_results_pca_znorm_11 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_Znorm_11, L, regression(l), 5, prior)]]
    
    print("PCA 11 done")

    plt.figure()
    plt.xlabel("Lambda")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Quadratic Logistic Regression Comparison")
    
    plt.plot(l_values, min_dcf_results_raw, label="minDCF(RAW)", color='blue')
    plt.plot(l_values, min_dcf_results_znorm, label="minDCF(Z-norm)",  color='green')
    plt.plot(l_values, min_dcf_results_pca_11, label="minDCF(PCA 11)",  color='orange')
    plt.plot(l_values, min_dcf_results_pca_znorm_11, label="minDCF(PCA 11 + Z-norm)",  color='red')
    
    plt.xlim(l_values[0], l_values[-1])
    plt.legend(loc='upper left')
    plt.savefig("Training/Logistic_Regression_Plot/Quadratic Logistic Regression Comparison.pdf")
    plt.close()
    
# Quadratic comparation plot 2 with PCA 10 and 9 (with and without znorm)
def quadratic_comparation_plot_2(D, L, prior):
    
    l_values = np.logspace(-5, 5, num=21)
    regression = QuadraticLogisticRegression
    
    D_Znorm = znorm(D)
    
    D_PCA_10, _ = PCA(D, 10)
    min_dcf_results_pca_10 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_10, L, regression(l), 5, prior)]]
    
    D_PCA_Znorm_10, _ = PCA(D_Znorm, 10)
    min_dcf_results_pca_znorm_10 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_Znorm_10, L, regression(l), 5, prior)]]
    
    print("PCA 10 done")
    
    D_PCA_9, _ = PCA(D, 9)
    min_dcf_results_pca_9 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_9, L, regression(l), 5, prior)]]
    
    D_PCA_Znorm_9, _  = PCA(D_Znorm, 9)
    min_dcf_results_pca_znorm_9 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for l in l_values 
               for SPost, Label in [kfold(D_PCA_Znorm_9, L, regression(l), 5, prior)]]
    
    print("PCA 9 done")
    
    plt.figure()
    plt.xlabel("Lambda")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Quadratic Logistic Regression Comparison 2")
    
    plt.plot(l_values, min_dcf_results_pca_10, label="minDCF(PCA 10)",  color='blue')
    plt.plot(l_values, min_dcf_results_pca_znorm_10, label="minDCF(PCA 10 + Z-norm)",  color='green')
    plt.plot(l_values, min_dcf_results_pca_9, label="minDCF(PCA 9)", color='orange')
    plt.plot(l_values, min_dcf_results_pca_znorm_9, label="minDCF(PCA 9 + Z-norm)",  color='red')
    
    plt.xlim(l_values[0], l_values[-1])
    plt.legend(loc='upper left')
    plt.savefig("Training/Logistic_Regression_Plot/Quadratic Logistic Regression Comparison 2.pdf")
    plt.close()   
    
def Quad_LR_diff_priors(D, L, ZNorm=False):
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
    if ZNorm:
        D = znorm(D)
        
    regression = QuadraticLogisticRegression(l)
    
    for pi_T, pi in priors:
        SPost, Label = kfold(D, L, regression, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF{'_znorm' if ZNorm else ''} (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")