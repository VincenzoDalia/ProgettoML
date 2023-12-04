import numpy as np
from Metrics.confusion_matrix import confusionMatrix


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