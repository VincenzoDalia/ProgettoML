import numpy as np
import scipy


####          GMM Functions         ####

def vrow(array):
    # trasforma un array in un vettore riga (1 riga, n colonne)
    return array.reshape((1, array.size))


def vcol(array):
    # trasforma un array in un vettore colonna (n righe, 1 colonna)
    return array.reshape((array.size, 1))


def calculate_mean_covariance(D):

    mean = vcol(D.mean(1))
    n_samples = D.shape[1]
    cov = np.dot(D-mean, (D-mean).T)/n_samples

    return mean, cov


def logpdf_GAU_ND(X, mu, C):

    M = X.shape[0]  # numero di features
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

def constr_eigenv(psi, gmm):
    for i in range(len(gmm)):
        covNew = gmm[i][2]
        U, s, _ = np.linalg.svd(covNew)
        s[s < psi] = psi
        gmm[i] = (gmm[i][0], gmm[i][1], np.dot(U, vcol(s) * U.T))

    return gmm


def tied_cov(gmm, vec, n):
    num_components = len(gmm)
    new_sigma = np.zeros_like(gmm[0][2])

    for idx in range(num_components):
        new_sigma += gmm[idx][2] * vec[idx]

    new_sigma /= n

    updated_gmm = [(component[0], component[1], new_sigma) for component in gmm]

    return updated_gmm


def EM(X, gmm, psi, type):
    llr_1 = None
    while True:
        num_components = len(gmm)

        logS = np.zeros((num_components, X.shape[1]))

        # E-STEP
        for idx in range(num_components):
            logS[idx, :] = logpdf_GAU_ND(X, gmm[idx][1], gmm[idx][2]) + np.log(
                gmm[idx][0]
            )
        logSMarginal = scipy.special.logsumexp(
            logS, axis=0
        )  # compute marginal densities
        SPost = np.exp(logS - logSMarginal)

        # M-STEP

        Z_vec = np.zeros(num_components)

        gmm_new = []
        for idx in range(num_components):
            gamma = SPost[idx, :]

            # update model parameters
            Z = gamma.sum()
            F = (vrow(gamma) * X).sum(1)
            S = np.dot(X, (vrow(gamma) * X).T)

            Z_vec[idx] = Z

            # new parameters
            mu = vcol(F / Z)
            sigma = S / Z - np.dot(mu, mu.T)
            w = Z / X.shape[1]

            gmm_new.append((w, mu, sigma))
        # END M-STEP

        if type == "GMM":
            gmm_new = gmm_new
        elif type == "Tied":
            gmm_new = tied_cov(gmm_new, Z_vec, X.shape[1])
        elif type == "Diagonal":
            pass
           #gmm_new = diagonal_cov(gmm_new, Z_vec, X.shape[1])
        elif type == "Tied-Diagonal":
            pass
           #gmm_new = TiedDiagonal_cov(gmm_new, Z_vec, X.shape[1])

        gmm_constr = constr_eigenv(psi, gmm_new)
        gmm = gmm_constr

        # stop criterion
        llr_0 = llr_1
        llr_1 = np.mean(logSMarginal)
        if llr_0 is not None and llr_1 - llr_0 < 1e-6:
            break

    return gmm


def LBG(iterations, X, gmm, alpha, psi, type):
    def update_gmm(gmm_start, alpha):
        new_gmm = []
        for g in gmm_start:
            start_w, start_mu, start_sigma = g

            U, s, Vt = np.linalg.svd(start_sigma)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha

            new_w = start_w / 2
            new_mu = start_mu
            new_sigma = start_sigma

            new_gmm.append((new_w, new_mu + d, new_sigma))
            new_gmm.append((new_w, new_mu - d, new_sigma))
        return new_gmm

    def initialize_gmm(gmm, X, type):
        if type == "GMM":
            return gmm
        elif type == "Tied":
            return tied_cov(gmm, [X.shape[1]], X.shape[1])
        elif type == "Diagonal":
            pass
           #return diagonal_cov(gmm, [X.shape[1]], X.shape[1])
        elif type == "Tied-Diagonal":
           pass
           #return TiedDiagonal_cov(gmm, [X.shape[1]], X.shape[1])

    def update_parameters(X, gmm, psi, type):
        gmm_constr = constr_eigenv(psi, gmm)
        return EM(X, gmm_constr, psi, type)

    gmm_start = initialize_gmm(gmm, X, type)
    gmm_start = update_parameters(X, gmm_start, psi, type)

    for _ in range(iterations):
        gmm_start = update_gmm(gmm_start, alpha)
        gmm_start = update_parameters(X, gmm_start, psi, type)

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





####          GMM Models         ####

class GMM:
    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi
        self.name = "GMM"
        
    def train(self, DTR, LTR, DTE, LTE, prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.prior = prior
        
        n_classes = np.unique(self.LTR).size
        
        # Inizializza una lista vuota per contenere gli oggetti LBG
        gmm = []

        # Itera attraverso le classi da 0 a num_classes - 1
        for classes in range(n_classes):
            # Seleziona le colonne corrispondenti alla classe corrente
            selected_columns = self.DTR[:, self.LTR == classes]
            
            # Calcola la media e la covarianza delle colonne selezionate
            class_mean_cov = calculate_mean_covariance(selected_columns)
            
            # Crea una lista contenente il valore 1 e la media/covarianza calcolata
            class_data = [1, *class_mean_cov]
            
            # Inizializza un oggetto LBG con i parametri appropriati
            lbg_object = LBG(
                self.iterations,
                selected_columns,
                [class_data],
                self.alpha,
                self.psi,
                self.name
            )
            
            # Aggiungi l'oggetto LBG alla lista
            gmm.append(lbg_object)
            
        self.gmm = gmm
        
    def calculate_scores(self):
        
        self.scores = gmm_scores(self.DTE, self.LTE, self.gmm)


class GMM_Tied:
    def __init__(self, iterations, alpha=0.1, psi=0.01):
        self.iterations = iterations
        self.alpha = alpha
        self.psi = psi
        self.name = "Tied"

    def train(self, DTR, LTR, DTE, LTE, eff_prior):
        self.DTR = DTR
        self.DTE = DTE
        self.LTR = LTR
        self.LTE = LTE

        num_classes = np.unique(self.LTR).size

        gmm = [
            LBG(
                self.iterations,
                self.DTR[:, self.LTR == classes],
                [[1, *calculate_mean_covariance(self.DTR[:, self.LTR == classes])]],
                self.alpha,
                self.psi,
                self.name,
            )
            for classes in range(num_classes)
        ]

        self.gmm = gmm

    def calculate_scores(self):
        self.scores = gmm_scores(self.DTE, self.LTE, self.gmm)

    