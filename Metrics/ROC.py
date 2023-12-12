import numpy as np
from Metrics.confusion_matrix import confusionMatrix
import matplotlib.pyplot as plt


def ROC(LTE, scores):
    
    thresholds = np.concatenate([scores, [-np.inf, np.inf]])
    thresholds.sort()
    
    FPR_axis = []
    TPR_axis = []
    
    for i in range(thresholds.shape[0]):
        
        predicted_labels = (scores > thresholds[i]).astype(int)
        conf_matrix = confusionMatrix(LTE, predicted_labels)
        
        (TN, FN), (FP, TP) = conf_matrix
        
        FPR = FP / (FP + TN)
        FNR = FN / (FN + TP)
        
        TPR = 1 - FNR
        
        FPR_axis.append(FPR)
        TPR_axis.append(TPR)
 
    return FPR_axis, TPR_axis



def plot_ROC_comparison(LR_scores, LR_LTE, SVM_scores, SVM_LTE, GMM_scores, GMM_LTE):
    
    LR_FPR, LR_TPR = ROC(LR_LTE, LR_scores)
    SVM_FPR, SVM_TPR = ROC(SVM_LTE, SVM_scores)
    GMM_FPR, GMM_TPR = ROC(GMM_LTE, GMM_scores)
    
        
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.plot(LR_FPR, LR_TPR, color="orange", label="LR")
    plt.plot(SVM_FPR, SVM_TPR, color="green", label="SVM")
    plt.plot(GMM_FPR, GMM_TPR, color="blue", label="GMM")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.savefig("Testing\ROC_comparison.pdf")
    plt.close()