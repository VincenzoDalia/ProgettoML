import numpy as np
from Functions.confusion_matrix import confusionMatrix


def DCF(pi, confMatrix, C_fn, C_fp, normalized):
        
        (TN, FN), (FP, TP) = confMatrix
        
        FNR = FN / (FN + TP)
        FPR = FP / (FP + TN)
        
        DCF = pi*C_fn*FNR + (1-pi)*C_fp*FPR
        
        
        if normalized == False:
            return DCF
        
        else:   
            B_dummy = min(pi * C_fn, (1 - pi) * C_fp)
            return DCF/B_dummy
        
    
def MIN_DCF(pi, C_fn, C_fp, LTE, scores):

    thresholds = np.concatenate([scores, [-np.inf, np.inf]])
    thresholds.sort()
    
    DCFs = []
    
    for i in range(thresholds.shape[0]):
        predicted_labels = (scores > thresholds[i]).astype(int)
        confusion_matrix = confusionMatrix(LTE, predicted_labels)
        
        temp_dcf = DCF(pi, confusion_matrix, C_fn, C_fp, True)
        
        DCFs.append(temp_dcf)
    
    return min(DCFs)