import numpy as np


def confusionMatrix(Real,Predicted):
    
    # n classes
    K = Real.max()+1 
    
    confMatrix = np.zeros((K,K), dtype=int)
    
    for i in range(Predicted.shape[0]):
        confMatrix[Predicted[i], Real[i]] += 1
        
    return confMatrix


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
    
    #min_dcf è già normalizzato
    
    #Aggiungo -inf e +inf a scores (per i due casi limite, tutto classificato come 0 o tutto come 1) e ordino
    thresholds = np.concatenate([scores, [-np.inf, np.inf]])
    thresholds.sort()
    
    DCFs = []
    
    for i in range(thresholds.shape[0]):
        predicted_labels = (scores > thresholds[i]).astype(int)
        confusion_matrix = confusionMatrix(LTE, predicted_labels)
        
        temp_dcf = DCF(pi, confusion_matrix, C_fn, C_fp, True)
        
        DCFs.append(temp_dcf)
    
    return min(DCFs)