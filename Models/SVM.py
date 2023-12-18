import numpy as np
import scipy.optimize as opt

### Functions for SVM ###

def mCol(v):
    return v.reshape((v.size, 1))

def mRow(array):
    return array.reshape((1,array.size))

def weighted_bounds(C, LTR, priors):
    emp = np.mean(LTR)  
    
    bounds = np.zeros_like(LTR, dtype=float)
    bounds[LTR == 1] = C * priors[1] / emp
    bounds[LTR == 0] = C * priors[0] / (1.0 - emp)
    
    return list(zip(np.zeros_like(LTR), bounds))

def lagrangian(alpha, H):
    
    ones =  np.ones(alpha.size) 
    
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
        Z = 2 * self.LTR - 1
        H = (Z[:, None] * Z) * G  
        
        alpha = np.zeros(self.LTR.size)
        
        bounds = weighted_bounds(self.C, self.LTR, [self.prior, 1-self.prior])

        (x, f, d) = opt.fmin_l_bfgs_b(lagrangian, x0=alpha, args=(H,), approx_grad=False, bounds=bounds, factr=1.0)
        
        self.w = np.sum( mCol(x) * mCol(Z) * D.T,  axis=0 )
        self.DTE = np.vstack([self.DTE, np.ones(self.DTE.shape[1]) * self.K])

    
    def calculate_scores(self):
        self.scores = np.dot(self.w.T, self.DTE) 
        self.scores = self.scores.reshape(-1)
        
    
class Polynomial_SVM:
    def __init__(self, K, constant, degree, C):
        self.K = K
        self.C = C
        self.degree = degree
        self.constant = constant
        
    def train(self, DTR, LTR, DTE, LTE, prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.prior = prior
        
        Z = 2*LTR -1    
        Z = mCol(Z)
        self.Z = Z
        
        polynomial_kernel_DTR = (np.dot(self.DTR.T,self.DTR) + self.constant)**self.degree
        
        polynomial_kernel_DTR = polynomial_kernel_DTR + self.K**2
        

        H = Z * Z.T * polynomial_kernel_DTR

        alpha = np.zeros(self.DTR.shape[1])
        
        bounds = weighted_bounds(self.C, self.LTR, [self.prior, 1-self.prior])

        (x, dual_objective, d) = opt.fmin_l_bfgs_b(lagrangian, x0=alpha, args=(H,), approx_grad=False, bounds=bounds, factr=1.0)
        
        self.alpha = x
        
    def calculate_scores(self):
        
        polynomial_kernel_DTE = (np.dot(self.DTR.T,self.DTE) + self.constant) ** self.degree + self.K**2
        self.scores = np.sum( mCol(self.alpha) * self.Z * polynomial_kernel_DTE, axis=0)
        
    

class Radial_SVM:
    def __init__(self, K, C, gamma):
        self.K = K
        self.C = C
        self.gamma = gamma
    
    def train(self, DTR, LTR, DTE, LTE, prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.prior = prior
        
        Z = 2*LTR -1    
        Z = mCol(Z)
        self.Z = Z
        
        radial_kernel_DTR = np.exp(-self.gamma * np.linalg.norm(self.DTR[:, :, np.newaxis] - self.DTR[:, np.newaxis, :], axis=0)**2) + self.K**2
        
        H = Z * Z.T * radial_kernel_DTR
        
        x0 = np.zeros(LTR.size)  
        bounds = weighted_bounds(self.C, self.LTR, [self.prior, 1-self.prior])
        
        (alpha_star, dual_objective, d) = opt.fmin_l_bfgs_b(lagrangian,args=(H,), approx_grad=False, x0=x0, bounds=bounds, factr=1.0)

        self.alpha = alpha_star


    
    def calculate_scores(self):
        
        radial_kernel_DTE = np.exp(-self.gamma * np.linalg.norm(self.DTR[:, :, np.newaxis] - self.DTE[:, np.newaxis, :], axis=0)**2) + self.K * self.K
        self.scores = np.sum(np.dot(self.alpha * mRow(self.Z), radial_kernel_DTE), axis=0)