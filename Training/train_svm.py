from Functions.load import *
from Functions.reshape_functions import *
from Functions.kfold import *
from Models.SVM import *
from Preprocessing.PCA import *
from Preprocessing.ZNorm import *
from Metrics.DCF import *
from Metrics.ROC import *
import matplotlib.pyplot as plt
from Metrics.BayesErr import *
from Calibration.Calibrate import *


###     SVM  GRAPHS    ###

    ###  Linear SVM      ###

# Grafico in cui fisso K e trovo C ottimale per Raw,ZNorm,PCA,PCA+ZNorm # (Lo eseguo per K=1)
def svm_comparation_plot(D, L, prior, K):
    C_values = np.logspace(-5, 5, num=15)
    svm = Linear_SVM
    
    min_dcf_results_raw = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for C in C_values 
               for SPost, Label in [kfold(D, L, svm(K,C), 5, prior)]]    
    
    print("RAW done")
    
    D_Znorm = znorm(D)
    min_dcf_results_znorm = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for C in C_values 
               for SPost, Label in [kfold(D_Znorm, L, svm(K,C), 5, prior)]]
    
    print("ZNorm done")
    
    D_PCA_11, _ = PCA(D, 11)
    min_dcf_results_pca_11 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for C in C_values 
               for SPost, Label in [kfold(D_PCA_11, L, svm(K,C), 5, prior)]]
    
    print("PCA 11 done")
    
    D_PCA_Znorm_11, _  = PCA(D_Znorm, 11)
    min_dcf_results_pca_znorm_11 = [MIN_DCF(0.5, 1, 1, Label, SPost)
               for C in C_values 
               for SPost, Label in [kfold(D_PCA_Znorm_11, L, svm(K,C), 5, prior)]]
    
    print("PCA 11 + ZNorm done")
    
    
    
    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Linear SVM Comparation Graph (K=1)")
    
    plt.plot(C_values, min_dcf_results_raw, label="minDCF(RAW)", color='blue')
    plt.plot(C_values, min_dcf_results_znorm, label="minDCF(Z-norm)",  color='green')
    plt.plot(C_values, min_dcf_results_pca_11, label="minDCF(PCA 11)",  color='orange')
    plt.plot(C_values, min_dcf_results_pca_znorm_11, label="minDCF(PCA 11 + Z-norm)",  color='red')
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend(loc='upper left')
    plt.savefig("Training/SVM_Plot/Linear SVM Comparation Graph (K=1).pdf")
    plt.close()


# Grafico RAW con K fissato, con pi che varia per trovare C
def linear_svm_raw_plot(D, L, prior, K):
    C_values = np.logspace(-5, 5, num=15)
    svm = Linear_SVM
    
    pi_values = [0.5, 0.1, 0.9]
    
    min_dcf_05 = [MIN_DCF(pi_values[0], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D, L, svm(K,C), 5, prior)]]
    
    min_dcf_01 = [MIN_DCF(pi_values[1], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D, L, svm(K,C), 5, prior)]]
    
    min_dcf_09 = [MIN_DCF(pi_values[2], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D, L, svm(K,C), 5, prior)]]
    
                    
    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Linear SVM - RAW (K=1, pi_T=0.5)")
    
    plt.plot(C_values, min_dcf_05, label="minDCF pi=0.5", color='blue')
    plt.plot(C_values, min_dcf_01, label="minDCF pi=0.1",  color='green')
    plt.plot(C_values, min_dcf_09, label="minDCF pi=0.9",  color='orange')
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend(loc='upper left')
    plt.savefig(f"Training/SVM_Plot/Linear SVM RAW (different pi).pdf")
    plt.close()
    

# Grafico ZNorm con K fissato, con pi che varia per trovare C
def linear_svm_znorm_plot(D, L,prior, K):
    C_values = np.logspace(-5, 5, num=15)
    svm = Linear_SVM
    
    D_Znorm = znorm(D)
    
    pi_values = [0.5, 0.1, 0.9]
    
    min_dcf_05 = [MIN_DCF(pi_values[0], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D_Znorm, L, svm(K,C), 5, prior)]]
    
    min_dcf_01 = [MIN_DCF(pi_values[1], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D_Znorm, L, svm(K,C), 5, prior)]]
    
    min_dcf_09 = [MIN_DCF(pi_values[2], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D_Znorm, L, svm(K,C), 5, prior)]]
    
    
    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Linear SVM - ZNorm (K=1, pi_T=0.5)")
    
    plt.plot(C_values, min_dcf_05, label="minDCF pi=0.5", color='blue')
    plt.plot(C_values, min_dcf_01, label="minDCF pi=0.1",  color='green')
    plt.plot(C_values, min_dcf_09, label="minDCF pi=0.9",  color='orange')
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend(loc='upper left')
    plt.savefig(f"Training/SVM_Plot/Linear SVM ZNorm (different pi).pdf")
    plt.close()
    
    

###  Polynomial SVM  ###
   

# Grafico RAW con K fissato, con pi che varia per trovare C
def polynomial_svm_raw_plot(D, L, prior, K):
    C_values = np.logspace(-5, 5, num=15)
    svm = Polynomial_SVM
    
    degree = 2
    constant = 1
    
    pi_values = [0.5, 0.1, 0.9]
    
    min_dcf_05 = [MIN_DCF(pi_values[0], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D, L, svm(K,constant,degree,C), 5, prior)]]
    
    min_dcf_01 = [MIN_DCF(pi_values[1], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D, L, svm(K,constant,degree,C), 5, prior)]]
    
    min_dcf_09 = [MIN_DCF(pi_values[2], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D, L, svm(K,constant,degree,C), 5, prior)]]
    
    
    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Polynomial SVM - RAW (K=1, d=2, c=1, pi_T=0.5)")
    
    plt.plot(C_values, min_dcf_05, label="minDCF pi=0.5", color='blue')
    plt.plot(C_values, min_dcf_01, label="minDCF pi=0.1",  color='green')
    plt.plot(C_values, min_dcf_09, label="minDCF pi=0.9",  color='orange')
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend(loc='upper left')
    plt.savefig(f"Training/SVM_Plot/Polynomial SVM RAW (different pi).pdf")
    plt.close()
    

# Grafico ZNorm con K fissato, con pi che varia per trovare C
def polynomial_svm_znorm_plot(D, L,prior, K):
    C_values = np.logspace(-5, 5, num=15)
    svm = Polynomial_SVM
    
    degree = 2
    constant = 1
    
    D_Znorm = znorm(D)
    
    pi_values = [0.5, 0.1, 0.9]
    
    min_dcf_05 = [MIN_DCF(pi_values[0], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D_Znorm, L, svm(K,constant,degree,C), 5, prior)]]
    
    min_dcf_01 = [MIN_DCF(pi_values[1], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D_Znorm, L, svm(K,constant,degree,C), 5, prior)]]
     
    min_dcf_09 = [MIN_DCF(pi_values[2], 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D_Znorm, L, svm(K,constant,degree,C), 5, prior)]]
    
    
    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Plynomial SVM - ZNorm (K=1, d=2, c=1, pi_T=0.5)")
    
    plt.plot(C_values, min_dcf_05, label="minDCF pi=0.5", color='blue')
    plt.plot(C_values, min_dcf_01, label="minDCF pi=0.1",  color='green')
    plt.plot(C_values, min_dcf_09, label="minDCF pi=0.9",  color='orange')
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend(loc='upper left')
    plt.savefig(f"Training/SVM_Plot/Polynomial SVM ZNorm (different pi).pdf")
    plt.close()



###  Radial SVM      ###

def radial_svm_raw_plot(D, L, prior, K):
    C_values = np.logspace(-5, 5, num=15)
    svm = Radial_SVM
    
    gamma = [0.001, 0.01, 0.1]
    
    min_dcf_0001 = [MIN_DCF(0.5, 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D, L, svm(K,C,gamma[0]), 5, prior)]]
    
    min_dcf_001 = [MIN_DCF(0.5, 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D, L, svm(K,C,gamma[1]), 5, prior)]]
    
    min_dcf_01 = [MIN_DCF(0.5, 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D, L, svm(K,C,gamma[2]), 5, prior)]]
    
    
    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Radial SVM - RAW (K=1, pi=0.5, pi_T=0.5)")
    
    plt.plot(C_values, min_dcf_0001, label="minDCF gamma=0.001", color='blue')
    plt.plot(C_values, min_dcf_001, label="minDCF gamma=0.01",  color='green')
    plt.plot(C_values, min_dcf_01, label="minDCF gamma=0.1",  color='orange')
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend(loc='upper left')
    plt.savefig(f"Training/SVM_Plot/Radial SVM RAW (different gamma).pdf")
    plt.close()
    
def radial_svm_znorm_plot(D, L, prior, K):
    C_values = np.logspace(-5, 5, num=15)
    svm = Radial_SVM
    
    D_Znorm = znorm(D)
    
    gamma = [0.001, 0.01, 0.1]
    
    min_dcf_0001 = [MIN_DCF(0.5, 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D_Znorm, L, svm(K,C,gamma[0]), 5, prior)]]
    
    min_dcf_001 = [MIN_DCF(0.5, 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D_Znorm, L, svm(K,C,gamma[1]), 5, prior)]]
    
    min_dcf_01 = [MIN_DCF(0.5, 1, 1, Label, SPost)
                for C in C_values
                for SPost, Label in [kfold(D_Znorm, L, svm(K,C,gamma[2]), 5, prior)]]
    
    
    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("Radial SVM - ZNorm (K=1, pi=0.5, pi_T=0.5)")
    
    plt.plot(C_values, min_dcf_0001, label="minDCF gamma=0.001", color='blue')
    plt.plot(C_values, min_dcf_001, label="minDCF gamma=0.01",  color='green')
    plt.plot(C_values, min_dcf_01, label="minDCF gamma=0.1",  color='orange')
    
    plt.xlim(C_values[0], C_values[-1])
    plt.legend(loc='upper left')
    plt.savefig(f"Training/SVM_Plot/Radial SVM ZNorm (different gamma).pdf")
    plt.close()

def SVM_candidate_train(D,L):
    D_Znorm = znorm(D)
    l = 0.1
    C = 5
    pi_T = 0.5
    
    svm = Radial_SVM(1, C, l)
    
    SPost, Label = kfold(D_Znorm, L, svm, 5, pi_T)
    
    return SPost, Label


def calibrated_SVM_dcf(D, L, prior):
    llr, Label = SVM_candidate_train(D, L)
    llr_cal, Label_cal = calibrate(llr, Label, 0.5)
    predicted_labels = optimal_bayes_decision(llr_cal, prior, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = MIN_DCF(prior, 1, 1, Label_cal, llr_cal)
    act_dcf = DCF(prior, conf_matrix, 1, 1, True)
    print(f"SVM (train) {prior}     min_dcf: {round(min_dcf, 3)}        act_dcf: {round(act_dcf, 3)}")
 

###     SVM  TABLES    ###

def SVM_diff_priors(D, L, ZNorm=False):
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

    if ZNorm:
        D = znorm(D)
    for pi_T, pi in priors:
        svm = Linear_SVM(1, C)
        SPost, Label = kfold(D, L,svm, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF{'_znorm' if ZNorm else ''} (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")

def RadKernBased_diff_priors(D, L, ZNorm=False):
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

    if ZNorm:
        D = znorm(D)
    for pi_T, pi in priors:
        svm = Radial_SVM(1, C, lbd)
        SPost, Label = kfold(D, L, svm, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF {'_znorm' if ZNorm else ''} (pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")

def Poly_SVM_diff_priors(D, L, ZNorm=False):
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
    
    if ZNorm:
        D = znorm(D)
    for pi_T, pi in priors:
        svm = Polynomial_SVM(1, 1, 2, C)
        SPost, Label = kfold(D, L, svm, 5, pi_T)
        res = MIN_DCF(pi, 1, 1, Label, SPost)
        print(f"min_DCF {'_znorm' if ZNorm else ''}(pi_T = {pi_T}, pi = {pi}) : {round(res, 3)}")
