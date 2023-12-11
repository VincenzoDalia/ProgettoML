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