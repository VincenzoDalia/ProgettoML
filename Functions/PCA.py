import numpy as np

def PCA(data_matrix, m, DTE=None):
    N = data_matrix.shape[1]
    mu = np.mean(data_matrix, axis=1, keepdims=True)
    DC = data_matrix - mu
    C = np.dot(DC, DC.T) / N
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, data_matrix)
    if DTE is not None:
        DTE = np.dot(P.T, DTE)
    return DP, DTE