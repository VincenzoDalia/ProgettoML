import numpy as np
import scipy.linalg  


def compute_Sb(dataset, label):
    num_features, num_samples = dataset.shape
    sb = np.zeros((num_features, num_features))
    class_means = []

    for i in range(2):
        class_samples = dataset[:, label == i]
        class_mean = np.mean(class_samples, axis=1, keepdims=True)
        class_means.append(class_mean)

    overall_mean = np.mean(dataset, axis=1, keepdims=True)
    dataset = dataset - overall_mean

    for i in range(2):
        class_size = np.sum(label == i)
        diff = class_means[i] - overall_mean
        sb += class_size * np.dot(diff, diff.T)

    Sb = (1 / num_samples) * sb
    return Sb

def compute_Sw(dataset, label):
    num_features, num_samples = dataset.shape
    mean_classes = []

    for i in range(2):
        class_samples = dataset[:, label == i]
        class_mean = np.mean(class_samples, axis=1, keepdims=True)
        mean_classes.append(class_mean)

    Sw = np.zeros((num_features, num_features))

    for i in range(2):
        class_size = np.sum(label == i)
        class_data = dataset[:, label == i]
        diff = class_data - mean_classes[i]
        Sw += class_size * (1 / class_data.shape[1]) * np.dot(diff, diff.T)

    Sw = (1 / num_samples) * Sw
    return Sw




# LDA Function 

def LDA(matrix, label, m):
    Sb = compute_Sb(matrix, label)
    Sw = compute_Sw(matrix, label)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    return W
