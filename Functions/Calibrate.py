import numpy as np
from Functions.kfold import kfold
from Models.LogisticRegression import BinaryLogisticRegression

def calibrate(scores, labels, prior):

    np.random.seed(4)

    indices = np.random.permutation(scores.size)
    shuffled_labels = labels[indices]
    shuffled_scores = scores[indices].reshape(1, scores.size)

    lr = BinaryLogisticRegression(0)

    scores_cv, labels_cv = kfold(shuffled_scores, shuffled_labels, lr, 5,  0.5)

    calibrated_scores = scores_cv - np.log(prior / (1 - prior))

    return calibrated_scores, labels_cv