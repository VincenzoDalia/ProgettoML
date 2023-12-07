import numpy as np
import scipy.optimize as opt

### Functions for SVM ###

def mCol(v):
    return v.reshape((v.size, 1))


def mRow(array):
    return array.reshape((1,array.size))


def acc_err_evaluate(Predicted_labels,Real_labels):
    
    result = np.array(Real_labels == Predicted_labels) # create an array of boolean with correct and uncorrect predictions

    acc = 100*(result.sum())/len(Real_labels) # summing an array of boolean returns the number of true values
    err = 100-acc
    
    return acc,err


def compute_duality_gap(w_hat_star, C, Z, D_hat, dual_obj):
  
    first_term = 0.5 * np.linalg.norm(w_hat_star) ** 2
    
    second_term = C * np.sum(np.maximum(0, 1 - ( Z.T * np.dot(w_hat_star.T, D_hat) )))
    
    primal_objective = first_term + second_term
    
    duality_gap = primal_objective + dual_obj
    
    return primal_objective,np.abs(duality_gap)

def weighted_bounds(C, LTR, priors):
    bounds = np.zeros(LTR.shape[0])
    emp = np.sum(LTR == 1) / LTR.shape[0]
    
    bounds[LTR == 1] = C * priors[1] / emp
    bounds[LTR == 0] = C * priors[0] / (1 - emp)
    
    return list(zip(np.zeros(LTR.shape[0]), bounds))


def lagrangian(alpha, H):
    
    ones =  np.ones(alpha.size) # 66,
    
    first_term = 0.5 * np.dot(np.dot(alpha.T, H), alpha)
    second_term = np.dot(alpha.T , mCol(ones))
    J_hat_D = first_term - second_term
    
    L_hat_D_gradient = np.dot(H, alpha) - ones 
    
    
    return J_hat_D, L_hat_D_gradient.flatten()




###     SVM Models   ###

class Linear_SVM:
    def __init__(self, K, C):
        self.K = K
        self.C = C
    
    def train(self, DTR, LTR, DTE, LTE, prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.prior = prior
        
        D = np.vstack([self.DTR, self.K * np.ones((1, self.DTR.shape[1]))])
        
        G = np.dot(D.T, D)
        
        #Z = self.LTR
        
        #Z[Z==0] = -1
        
        #Z = mCol(Z)
        
        #H = Z * Z.T * G
        
        Z = 2 * self.LTR - 1
        H = (Z[:, None] * Z) * G  

        # Compute Dual solution
        alpha = np.zeros(self.LTR.size)
        
        
        #bounds_list = [(0,self.C)] * self.LTR.size  # va bene solo nel lab
        
        bounds = weighted_bounds(self.C, self.LTR, [self.prior, 1-self.prior])
        
        
        
        (x, f, d) = opt.fmin_l_bfgs_b(lagrangian, x0=alpha, args=(H,), approx_grad=False, bounds=bounds, factr=1.0)
        
        
        """ # Recover Primal solution
        w_hat_star = np.sum( mCol(x) * mCol(Z) * D.T,  axis=0 )
        
        # Extract terms and compute scores
        w_star = w_hat_star[0:-1] 
        b_star = w_hat_star[-1] * self.K

        scores = np.dot(w_star.T, DTE) + b_star
        
        primal_obj,duality_gap = compute_duality_gap(w_hat_star, self.C, Z, D,f)
        dual_obj = f """
        
        self.w = np.sum( mCol(x) * mCol(Z) * D.T,  axis=0 )
        self.DTE = np.vstack([self.DTE, np.ones(self.DTE.shape[1]) * self.K])

    
    def calculate_scores(self):
        self.scores = np.dot(self.w.T, self.DTE) 
        self.scores = self.scores.reshape(-1)
        
    
    
    
    
    
    
    
    
    
    
    
    
class Radial_SVM:
    def __init__(self, K, C, gamma):
        self.K = K
        self.C = C
        self.gamma = gamma
        
class Polynomial_SVM:
    def __init__(self, K, C, d, c):
        self.K = K
        self.C = C
        self.d = d
        self.c = c
    