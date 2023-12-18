import numpy as np
import scipy


####          GMM Functions         ####

def vrow(array):
    return array.reshape((1, array.size))


def vcol(array):
    return array.reshape((array.size, 1))


def calculate_mean_covariance(D):

    mean = vcol(D.mean(1))
    n_samples = D.shape[1]
    cov = np.dot(D-mean, (D-mean).T)/n_samples

    return mean, cov


def logpdf_GAU_ND(X, mu, C):

    M = X.shape[0]  
    first_term = -0.5 * M * np.log(2*np.pi)

    log_det = np.linalg.slogdet(C)[1]
    second_term = -0.5 * log_det
    X_c = X - mu
    C_inv = np.linalg.inv(C)

    third_term = -0.5 * np.dot(np.dot(X_c.T, C_inv), X_c)
    third_term = third_term.diagonal()

    logpdf = first_term + second_term + third_term

    return logpdf


def logpdf_gmm(X, gmm):
    log_probs = []
    for component in gmm:
        log_weight = np.log(component[0])
        log_prob = logpdf_GAU_ND(X, component[1], component[2]) + log_weight
        log_probs.append(log_prob)
    
    log_probs = np.vstack(log_probs)
    return np.logaddexp.reduce(log_probs, axis=0)


def tied_covariance(gmm, vec, n):
    sigmas = np.array([component[2] for component in gmm])
    weights = np.array(vec) / n
    new_sigma = np.average(sigmas, weights=weights, axis=0)

    updated_gmm = [(w, mu, new_sigma) for w, mu, _ in gmm]

    return updated_gmm

def diagonal_covariance(gmm, vec, n):
    return [(g[0], g[1], np.diag(np.diag(g[2]))) for g in gmm]


def tiedDiagonal_covariance(gmm, vec, n):
    tied_diagonal_gmm = diagonal_covariance(tied_covariance(gmm, vec, n), vec, n)
    return tied_diagonal_gmm




def constrain_eigen(psi, gmm):
    for i, (mean, weight, covNew) in enumerate(gmm):
        U, s, Vt = np.linalg.svd(covNew, full_matrices=False)
        s[s < psi] = psi
        gmm[i] = (mean, weight, U @ np.diag(s) @ Vt)

    return gmm

def EM(X, gmm, psi, type):
    
    llr_1 = None
    num_components = len(gmm)
    X_shape_1 = X.shape[1]
    logS = np.zeros((num_components, X_shape_1))
    Z_vec = np.zeros(num_components)

    while True:
        # E-STEP
        for idx in range(num_components):
            logS[idx, :] = logpdf_GAU_ND(X, gmm[idx][1], gmm[idx][2]) + np.log(gmm[idx][0])
        logSMarginal = np.log(np.sum(np.exp(logS - np.max(logS, axis=0, keepdims=True)), axis=0)) + np.max(logS, axis=0, keepdims=True) 
        SPost = np.exp(logS - logSMarginal)

        # M-STEP
        gmm_new = []
        for idx in range(num_components):
            gamma = SPost[idx, :]
            Z = gamma.sum()
            F = (vrow(gamma) * X).sum(1)
            S = np.dot(X, (vrow(gamma) * X).T)
            Z_vec[idx] = Z
            mu = vcol(F / Z)
            sigma = S / Z - np.dot(mu, mu.T)
            w = Z / X_shape_1
            gmm_new.append((w, mu, sigma))

        if type == "Tied":
            gmm_new = tied_covariance(gmm_new, Z_vec, X_shape_1)
        elif type == "Diagonal":
            gmm_new = diagonal_covariance(gmm_new, Z_vec, X_shape_1)
        elif type == "Tied-Diagonal":
            gmm_new = tiedDiagonal_covariance(gmm_new, Z_vec, X_shape_1)

        gmm = constrain_eigen(psi, gmm_new)

        llr_0 = llr_1
        llr_1 = np.mean(logSMarginal)
        if llr_0 is not None and np.abs(llr_1 - llr_0) < 1e-6:
            break

    return gmm
 
def LBG(iterations, X, gmm, alpha, psi, type):
        
    def update_gmm(gmm_start, alpha):
        new_gmm = []
        for start_w, start_mu, start_sigma in gmm_start:
            U, s, _ = np.linalg.svd(start_sigma)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            new_w = start_w / 2

            new_gmm.extend([(new_w, start_mu + d, start_sigma), 
                            (new_w, start_mu - d, start_sigma)])
        return new_gmm

    def initialize_gmm(gmm, X, type):
        if type == "GMM":
            return gmm
        elif type == "Tied":
            return tied_covariance(gmm, [X.shape[1]], X.shape[1])
        elif type == "Diagonal":
            return diagonal_covariance(gmm, [X.shape[1]], X.shape[1])
        elif type == "Tied-Diagonal":
            return tiedDiagonal_covariance(gmm, [X.shape[1]], X.shape[1])

    gmm_start = initialize_gmm(gmm, X, type)
    gmm_start = constrain_eigen(psi, gmm_start)
    gmm_start = EM(X, gmm_start, psi, type)

    for _ in range(iterations):
        gmm_start = update_gmm(gmm_start, alpha)
        gmm_start = constrain_eigen(psi, gmm_start)
        gmm_start = EM(X, gmm_start, psi, type)

    return gmm_start

def gmm_scores(D, L, gmm):
    unique_labels = np.unique(L)
    num_classes = unique_labels.size
    num_features = D.shape[1]

    scores = np.zeros((num_classes, num_features))
    for i, label in enumerate(unique_labels):
        class_scores = logpdf_gmm(D, gmm[label])
        scores[i] = np.exp(class_scores)

    llr = np.log(scores[1] / scores[0])

    return llr



####          GMM Model         ####

class GMM:
    def __init__(self, iterations, name, alpha=0.1, psi=0.01):
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi
        self.name = name  
        
    def train(self, DTR, LTR, DTE, LTE, prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.prior = prior
        
        n_classes = np.unique(self.LTR).size
        
        gmm = []

        for classes in range(n_classes):
            
            selected_columns = self.DTR[:, self.LTR == classes]
            
            class_mean_cov = calculate_mean_covariance(selected_columns)
            
            class_data = [1, *class_mean_cov]
            
            lbg_object = LBG(
                self.iterations,
                selected_columns,
                [class_data],
                self.alpha,
                self.psi,
                self.name
            )
            
            gmm.append(lbg_object)
            
        self.gmm = gmm
        
    def calculate_scores(self):
        
        self.scores = gmm_scores(self.DTE, self.LTE, self.gmm)

