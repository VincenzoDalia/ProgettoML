import numpy as np


def kfold(Data, Labels, model, k, prior, seed=3):
    
    N = Data.shape[1]
    fold_size = N//k #calcolo quanti samples ci sono in ogni fold al massimo.
    
    # Permuta casualmente gli indici dei dati
    np.random.seed(seed)
    indices = np.random.permutation(N)  
    Labels = Labels[indices]
    
    scores = []
    
    for i in range(k):
        
        test_indices = indices[i * fold_size: (i + 1) * fold_size] 
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
        
        if (Data.shape[1] % k) != 0 and i==k-1:
            test_indices = np.concatenate((test_indices, indices[k * fold_size:]))
        
        X_train = Data[:, train_indices]
        y_train = Labels[train_indices]
        X_test = Data[:, test_indices]
        y_test = Labels[test_indices]
        
        model.train(X_train, y_train, X_test, y_test, prior) 
        
        model.calculate_scores()
        partial_scores = model.scores
        
        
        scores.append(partial_scores)
        
        
    SPost = np.hstack(scores)  
    
    return SPost,Labels
        
    