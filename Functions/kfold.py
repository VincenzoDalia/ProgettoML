import numpy as np

def kfold(D, L, model, k, eff_prior=None, seed=3):
    SPost_partial = []
    
    N = D.shape[1]
    fold_size = N // k  
    remainder = N % k  
    
    np.random.seed(seed)
    idx = np.random.permutation(N)

    Label = L[idx]

    start_idx = 0
    for i in range(k):
        
        current_fold_size = fold_size + 1 if i < remainder else fold_size

        test_indices = idx[start_idx: start_idx + current_fold_size]
        train_indices = np.concatenate((idx[:start_idx], idx[start_idx + current_fold_size:]))

        start_idx += current_fold_size

        model.train(
            D[:, train_indices],
            L[train_indices],
            D[:, test_indices],
            L[test_indices],
            eff_prior,
        )
        model.calculate_scores()
        scores = model.scores

        SPost_partial.append(scores)

    S = np.hstack(SPost_partial)

    return S, Label