import numpy as np


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
