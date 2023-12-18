import numpy as np


def confusionMatrix(Real,Predicted):
    
    K = Real.max()+1 
    
    confMatrix = np.zeros((K,K), dtype=int)
    
    for i in range(Predicted.shape[0]):
        confMatrix[Predicted[i], Real[i]] += 1
        
    return confMatrix