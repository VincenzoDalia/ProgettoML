import numpy as np
import sklearn.datasets
import scipy.optimize


def polynomial_transformation(DTR, DTE):
    n_train = DTR.shape[1]
    n_eval = DTE.shape[1]
    n_f = DTR.shape[0]

    quad_dtr = np.zeros((n_f ** 2 + n_f, n_train))
    quad_dte = np.zeros((n_f ** 2 + n_f, n_eval))

    quad_dtr[:n_f ** 2, :] = (DTR.reshape(n_f, 1, n_train) * DTR.reshape(1, n_f, n_train)).reshape(n_f**2, -1)
    quad_dtr[n_f ** 2:, :] = DTR

    quad_dte[:n_f ** 2, :] = (DTE.reshape(n_f, 1, n_eval) * DTE.reshape(1, n_f, n_eval)).reshape(n_f**2, -1)
    quad_dte[n_f ** 2:, :] = DTE

    return quad_dtr, quad_dte



class BinaryLogisticRegression:
    def __init__(self, l):
        self.l = l
        
    def logreg_obj(self, v):
        
        w, b = v[:-1], v[-1]

        priors = np.array([self.eff_prior, 1 - self.eff_prior])

        first_elem = (self.l / 2) * np.linalg.norm(w) ** 2

        loss_funct = []
        classes = np.unique(self.LTR)

        for i in classes:
            indices = np.where(self.LTR == i)[0]
            const_class = priors[i] / len(indices)

            z = 2 * self.LTR[indices] - 1

            exp_term = np.logaddexp(0, -z * (np.dot(w.T, self.DTR[:, indices]) + b))

            loss_funct.append(const_class * np.sum(exp_term))

        return first_elem + loss_funct[0] + loss_funct[1]

    def train(self, DTR, LTR, DTE, LTE, eff_prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.eff_prior = eff_prior
        self.LTE = LTE
        
        x0 = np.zeros(self.DTR.shape[0] + 1)
        
        x, _ , _ = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0, approx_grad = True, factr=10000000.0, maxfun=20000)
        
        self.w, self.b = x[:-1], x[-1]
        
        
    def calculate_scores(self):
       self.scores = np.dot(self.w.T, self.DTE) + self.b
       
           
class QuadraticLogisticRegression:
    def __init__(self, l):
        self.l = l
        
    def logreg_obj(self, v):
        
        w, b = v[:-1], v[-1]

        priors = np.array([self.eff_prior, 1 - self.eff_prior])

        first_elem = (self.l / 2) * np.linalg.norm(w) ** 2

        loss_funct = []
        classes = np.unique(self.LTR)

        for i in classes:
            indices = np.where(self.LTR == i)[0]
            const_class = priors[i] / len(indices)

            z = 2 * self.LTR[indices] - 1

            exp_term = np.logaddexp(0, -z * (np.dot(w.T, self.DTR[:, indices]) + b))

            loss_funct.append(const_class * np.sum(exp_term))

        return first_elem + loss_funct[0] + loss_funct[1]

    def train(self, DTR, LTR, DTE, LTE, eff_prior):
        quad_DTR, quad_DTE = polynomial_transformation(DTR, DTE)
        self.DTR = quad_DTR
        self.LTR = LTR
        self.DTE = quad_DTE
        self.eff_prior = eff_prior
        self.LTE = LTE
        
        x0 = np.zeros(self.DTR.shape[0] + 1)
        
        x, _ , _ = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0, approx_grad = True, factr=10000000.0, maxfun=20000)
        
        self.w, self.b = x[:-1], x[-1]
        
        
    def calculate_scores(self):
       self.scores = np.dot(self.w.T, self.DTE) + self.b
       
    