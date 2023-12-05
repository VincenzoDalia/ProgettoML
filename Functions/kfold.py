import numpy as np


###   KFOLD GINO
""" 
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
         
     """
""" ###   KFOLD MARIO
def kfold(Data, Labels, model, k, prior, seed=4):
    
    N = Data.shape[1]
    fold_size = N//k #calcolo quanti samples ci sono in ogni fold al massimo.
    
    # Permuta casualmente gli indici dei dati
    np.random.seed(seed)
    indices = np.random.permutation(N)
    
    Labels = Labels[indices]
    
    scores = []
    
    for i in range(k):
        
        test_indices = indices[i * fold_size: (i + 1) * fold_size] #i*fold_size indica l'indice iniziale per considerare il fold, (i+1)*fold size indica l'indice finale(non incluso)
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
        
        if (Data.shape[1] % k) != 0 and i==k-1:
            test_indices = np.concatenate((test_indices, indices[k * fold_size:]))
        
        X_train = Data[:, train_indices]
        y_train = Labels[train_indices]
        X_test = Data[:, test_indices]
        y_test = Labels[test_indices]
        
        model.train(X_train, y_train, X_test, y_test, prior)
        model.calculate_scores()
        SPost = model.scores
        
        scores.append(SPost)
        
    
    SPost = np.hstack(scores)  
    
    
    return SPost,Labels
 """

def kfold(D, L, model, k, eff_prior=None, seed=3):
    SPost_partial = []
    folds = []

    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    Label = L[idx]

    fold_size = D.shape[1] // k

    # Divide indices into k-folds
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        folds.append(idx[start:end])

    # If the number of samples is not divisible by K, add the remaining samples to the last fold
    if D.shape[1] % k != 0:
        folds[-1] = np.concatenate((folds[-1], idx[k * fold_size :]))

    # Perform Cross validation
    for i in range(k):
        # Choose the i-th fold as the validation fold
        validation_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        model.train(
            D[:, train_indices],
            L[train_indices],
            D[:, validation_indices],
            L[validation_indices],
            eff_prior,
        )
        model.calculate_scores()
        scores = model.scores

        SPost_partial.append(scores)
        # print("end fold:",i)

    S = np.hstack(SPost_partial)

    return S, Label
