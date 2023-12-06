import numpy as np

def kfold(D, L, model, k, eff_prior=None, seed=3):
    SPost_partial = []
    
    N = D.shape[1]
    fold_size = N // k  # Calcolo quanti samples ci sono in ogni fold al massimo.
    remainder = N % k  # Calcolo il numero di elementi di resto
    
    np.random.seed(seed)
    idx = np.random.permutation(N)

    Label = L[idx]

    start_idx = 0
    for i in range(k):
        # Calcolo la dimensione del fold corrente, considerando gli elementi di resto
        current_fold_size = fold_size + 1 if i < remainder else fold_size

        test_indices = idx[start_idx: start_idx + current_fold_size]
        train_indices = np.concatenate((idx[:start_idx], idx[start_idx + current_fold_size:]))

        # Aggiorno l'indice di partenza per il prossimo fold
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
        # print("end fold:",i)

    S = np.hstack(SPost_partial)

    return S, Label