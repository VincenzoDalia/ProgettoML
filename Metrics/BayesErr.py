import numpy as np
import matplotlib.pyplot as plt
from Metrics.confusion_matrix import *
from Metrics.DCF import *



def optimal_bayes_decision(llr,pi,Cf_n,Cf_p):

    x = pi*Cf_n/((1-pi)*Cf_p)
    t = -np.log(x)

    predicted_labels = (llr > t).astype(int)
    
    return predicted_labels




def Bayes_Error(L, scores, filename):
    
    effPriorLogOdds = np.linspace(-3, 3, 50)    

    DCFs = []
    min_DCFs = []
    
    # calcolo effective prior
    for p in effPriorLogOdds:
        
        pi = 1/(1+np.exp(-p))

        temp_min_DCF = MIN_DCF(pi, 1, 1, L, scores)
        min_DCFs.append(temp_min_DCF)

        temp_predicted_labels = optimal_bayes_decision(scores, pi, 1, 1)
        temp_conf_matrix = confusionMatrix(L,temp_predicted_labels)
        temp_DCF = DCF(pi, temp_conf_matrix, 1, 1, True)    
        DCFs.append(temp_DCF)
        
    plt.figure(figsize=(8,6))
    plt.plot(effPriorLogOdds, DCFs, label="DCF", color="r")
    plt.plot(effPriorLogOdds, min_DCFs, label="min DCF", color="b")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel("Prior Log-odds")
    plt.ylabel("DCF")
    plt.title(f"Bayes error plot {filename}")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Calibration/{filename}_Bayes_Error.pdf")
    plt.close()
    
    
    
def plot_Bayes_Error_Comparison(L_LR, scores_LR, L_SVM, scores_SVM, L_GMM, scores_GMM, filename):
    
    effPriorLogOdds = np.linspace(-3, 3, 50)    

    DCFs_LR = []
    min_DCFs_LR = []
    
    DCFs_SVM = []
    min_DCFs_SVM = []
    
    DCFs_GMM = []
    min_DCFs_GMM = []
    
    #Logistic Regression
    for p in effPriorLogOdds:
        
        pi = 1/(1+np.exp(-p))

        temp_min_DCF = MIN_DCF(pi, 1, 1, L_LR, scores_LR)
        min_DCFs_LR.append(temp_min_DCF)

        temp_predicted_labels = optimal_bayes_decision(scores_LR, pi, 1, 1)
        temp_conf_matrix = confusionMatrix(L_LR,temp_predicted_labels)
        temp_DCF = DCF(pi, temp_conf_matrix, 1, 1, True)    
        DCFs_LR.append(temp_DCF)
        
    #SVM
    for p in effPriorLogOdds:
        
        pi = 1/(1+np.exp(-p))

        temp_min_DCF = MIN_DCF(pi, 1, 1, L_SVM, scores_SVM)
        min_DCFs_SVM.append(temp_min_DCF)

        temp_predicted_labels = optimal_bayes_decision(scores_SVM, pi, 1, 1)
        temp_conf_matrix = confusionMatrix(L_SVM,temp_predicted_labels)
        temp_DCF = DCF(pi, temp_conf_matrix, 1, 1, True)    
        DCFs_SVM.append(temp_DCF)
        
    #GMM
    for p in effPriorLogOdds:
        
        pi = 1/(1+np.exp(-p))

        temp_min_DCF = MIN_DCF(pi, 1, 1, L_GMM, scores_GMM)
        min_DCFs_GMM.append(temp_min_DCF)

        temp_predicted_labels = optimal_bayes_decision(scores_GMM, pi, 1, 1)
        temp_conf_matrix = confusionMatrix(L_GMM,temp_predicted_labels)
        temp_DCF = DCF(pi, temp_conf_matrix, 1, 1, True)    
        DCFs_GMM.append(temp_DCF)
        
    plt.figure(figsize=(8,6))
    plt.plot(effPriorLogOdds, DCFs_LR, label="DCF LR", color="r", linestyle="-")
    plt.plot(effPriorLogOdds, min_DCFs_LR, label="min DCF LR", color="r", linestyle="--")
    plt.plot(effPriorLogOdds, DCFs_SVM, label="DCF SVM", color="b", linestyle="-")
    plt.plot(effPriorLogOdds, min_DCFs_SVM, label="min DCF SVM", color="b", linestyle="--")
    plt.plot(effPriorLogOdds, DCFs_GMM, label="DCF GMM", color="g", linestyle="-")
    plt.plot(effPriorLogOdds, min_DCFs_GMM, label="min DCF GMM", color="g", linestyle="--")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel("Prior Log-odds")
    plt.ylabel("DCF")
    plt.title(f"Bayes error plot {filename}")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Testing/{filename}_Bayes_Error_Comparison.pdf")
    plt.close()