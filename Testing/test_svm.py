from Functions.load import *
from Functions.reshape_functions import *
from Functions.kfold import *
from Models.SVM import *
from Preprocessing.PCA import *
from Metrics.DCF import *
from Metrics.ROC import *
from Preprocessing.ZNorm import *
import matplotlib.pyplot as plt
from Metrics.BayesErr import *
from Calibration.Calibrate import *

def SVM_candidate_test(DTR,LTR,DTE,LTE):
    
    DTR_znorm = znorm(DTR)
    DTE_znorm = znorm(DTE)
    
    gamma = 0.1
    C = 5
    pi_T = 0.5
    K = 1
    
    svm = Radial_SVM(K, C, gamma)
    
    svm.train(DTR_znorm, LTR, DTE_znorm, LTE, pi_T)
    svm.calculate_scores()
    
    scores = svm.scores
    
    return scores, LTE


def calibrated_LR_test_dcf(DTR, LTR, DTE, LTE, prior):
    
    llr, Label = SVM_candidate_test(DTR, LTR, DTE, LTE, prior)
    
    llr_cal, Label_cal = calibrate(llr, Label, 0.5)
    predicted_labels = optimal_bayes_decision(llr_cal, prior, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = MIN_DCF(prior, 1, 1, Label_cal, llr_cal)
    act_dcf = DCF(prior, conf_matrix, 1, 1, True)
    
    print(f"Radial SVM (Test) {prior}     min_dcf: {round(min_dcf, 3)}        act_dcf: {round(act_dcf, 3)}")