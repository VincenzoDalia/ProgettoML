import numpy as np
import sklearn.datasets
import scipy



######           MVG model          ######



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

def acc_err_evaluate(Predicted_labels, Real_labels):

    # create an array of boolean with correct and uncorrect predictions
    result = np.array(Real_labels == Predicted_labels)
    # summing an array of boolean returns the number of true values
    acc = 100*(result.sum())/len(Real_labels)
    err = 100-acc

    return acc, err

def evaluate_accuracy(Posterior_prob,Real_labels):
    
    Predicted_labels = np.argmax(Posterior_prob,0)
    result = np.array([Real_labels[i] == Predicted_labels[i] for i in range(len(Real_labels))]) # create an array of boolean with correct and uncorrect predictions

    acc = 100*(result.sum())/len(Real_labels) # summing an array of boolean returns the number of true values
    
    return acc



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
    

class LogGaussian_Classifier:
    
    def __init__(self):
        self.name = "LogGaussian"
    
    
    def train(self,DTR,LTR,DTE,LTE,prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.prior = prior
        
        S = []
    
        #Calcolo la probabilità condizionata per ogni classe e per ogni elemento del test set
        for i in range(self.LTR.max()+1):
            
            DTR_i = self.DTR[:, self.LTR == i]
            mean_i, cov_i = calculate_mean_covariance(DTR_i)
            class_conditional_prob = logpdf_GAU_ND(self.DTE, mean_i, cov_i)
            
            #Uso vrow per trasformare il vettore in una matrice riga (1 riga, n colonne) 
            #per poterlo concatenare con le altre matrici riga dentro S
            S.append(vrow(class_conditional_prob))
            
        S = np.vstack(S)
        
        prior = np.ones(S.shape)*[[self.prior], [1-self.prior]]
        
        #Al posto di moltiplicare per prior, sommo il logaritmo di prior
        #sfruttando la proprietà log(a*b) = log(a) + log(b) 
        logSJoint = S + np.log(prior)
        
        logSMargin = vrow(scipy.special.logsumexp(logSJoint, axis=0))
        
        logSPost = logSJoint - logSMargin
        self.logSPost = logSPost
    
    def calculate_scores(self):
        
        scores = self.logSPost[1,:] - self.logSPost[0,:] - np.log(self.prior/(1-self.prior))   
        self.scores = scores     
 

class NBGaussian_Classifier:
    
    def __init__(self):
        self.name = "NBGaussian"
        
    def train(self,DTR,LTR,DTE,LTE,prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.prior = prior
        
        S = []
    
        for i in range(self.LTR.max()+1):
            
            DTR_i = self.DTR[:, self.LTR == i]
            mean_i, cov_i = calculate_mean_covariance(DTR_i)
            
            #in questo caso la covarianza è diagonale e quindi devo moltiplicarla per la matrice identità
            identity = np.identity(cov_i.shape[0])
            cov_i = cov_i*identity

            class_conditional_prob = logpdf_GAU_ND(self.DTE, mean_i, cov_i)

            S.append(vrow(class_conditional_prob))
            
        S = np.vstack(S)
        
        prior = np.ones(S.shape)*[[self.prior], [1-self.prior]]
        
        #Al posto di moltiplicare per prior, sommo il logaritmo di prior
        #sfruttando la proprietà log(a*b) = log(a) + log(b) 
        logSJoint = S + np.log(prior)
        logSMargin = vrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMargin
        
        self.logSPost = logSPost
        
    
    def calculate_scores(self):
        
        scores = self.logSPost[1,:] - self.logSPost[0,:] - np.log(self.prior/(1-self.prior))   
        self.scores = scores    


class TiedGaussian_Classifier:
    
    def __init__(self):
        self.name = "TiedGaussian"
        
    def train(self,DTR,LTR,DTE,LTE,prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.prior = prior
        
        
        S = []
    
        #calcolo la matrice di covarianza condivisa
        tied_cov = 0
        n_tot_samples = self.DTR.shape[1] #numero di campioni TOTALI
        
        for i in range(self.LTR.max()+1):
            
            DTR_i = self.DTR[:, self.LTR == i]
            n_class_samples = DTR_i.shape[1] #numero di campioni per classe  
            cov_i = calculate_mean_covariance(DTR_i)[1]
            tied_cov += n_class_samples*cov_i #moltiplico la covarianza per il numero di campioni per classe per annullare l'effetto della cardinalità delle classi
            
        tied_cov = tied_cov/n_tot_samples #covarianza condivisa
        
        #Calcolo la probabilità condizionata per ogni classe e per ogni elemento del test set
        for i in range(LTR.max()+1):
            
            DTR_i = DTR[:, LTR == i]
            mean_i, cov_i = calculate_mean_covariance(DTR_i)
            class_conditional_prob = logpdf_GAU_ND(DTE, mean_i, tied_cov)
            
            S.append(vrow(class_conditional_prob))
            
        S = np.vstack(S)
        
        prior = np.ones(S.shape)*[[self.prior], [1-self.prior]]
        
        logSJoint = S + np.log(prior)
        
        logSMargin = vrow(scipy.special.logsumexp(logSJoint, axis=0))
        
        logSPost = logSJoint - logSMargin
        
        self.logSPost = logSPost
    
    def calculate_scores(self):
        
        scores = self.logSPost[1,:] - self.logSPost[0,:] - np.log(self.prior/(1-self.prior))   
        self.scores = scores
        
 
class TiedNBGaussian_Classifier:
    
    def __init__(self):
        self.name = "TiedNBGaussian"
        
    def train(self,DTR,LTR,DTE,LTE,prior):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        self.prior = prior
        
        S = []
    
        #calcolo la matrice di covarianza condivisa
        tied_cov = 0
        n_tot_samples = self.DTR.shape[1] #numero di campioni TOTALI
        
        for i in range(self.LTR.max()+1):
            
            DTR_i = self.DTR[:, self.LTR == i]
            n_class_samples = DTR_i.shape[1] #numero di campioni per classe  
            cov_i = calculate_mean_covariance(DTR_i)[1]
            tied_cov += n_class_samples*cov_i #moltiplico la covarianza per il numero di campioni per classe per annullare l'effetto della cardinalità delle classi
            
        
        tied_cov = tied_cov/n_tot_samples #covarianza condivisa
        
        #Assunzione Naive Bayes: covarianza diagonale
        identity = np.identity(cov_i.shape[0])
        tied_cov = tied_cov*identity
        
        #Calcolo la probabilità condizionata per ogni classe e per ogni elemento del test set
        for i in range(LTR.max()+1):
            
            DTR_i = DTR[:, LTR == i]
            mean_i, cov_i = calculate_mean_covariance(DTR_i)
            class_conditional_prob = logpdf_GAU_ND(DTE, mean_i, tied_cov)
            
            #Uso vrow per trasformare il vettore in una matrice riga (1 riga, n colonne) 
            #per poterlo concatenare con le altre matrici riga dentro S
            S.append(vrow(class_conditional_prob))
            
        S = np.vstack(S)
        
        prior = np.ones(S.shape)*[[self.prior], [1-self.prior]]
        
        logSJoint = S + np.log(prior)
        
        logSMargin = vrow(scipy.special.logsumexp(logSJoint, axis=0))
        
        logSPost = logSJoint - logSMargin
        
        self.logSPost = logSPost
        
        
    def calculate_scores(self):
        
        scores = self.logSPost[1,:] - self.logSPost[0,:] - np.log(self.prior/(1-self.prior))   
        self.scores = scores

