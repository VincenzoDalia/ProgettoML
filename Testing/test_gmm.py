from Functions.load import *
from Functions.reshape_functions import *
from Functions.kfold import *
from Models.GMM import *
from Preprocessing.PCA import *
from Metrics.DCF import *
from Metrics.ROC import *
from Preprocessing.ZNorm import *
import matplotlib.pyplot as plt
from Metrics.BayesErr import *
from Calibration.Calibrate import *

def GMM_candidate_test(DTR,LTR,DTE,LTE):
    
    #4 components --> 2 iterations
    gmm = GMM(2,"GMM")
    
    DTR_pca, _ = PCA(DTR, 11)
    DTE_pca, _ = PCA(DTE, 11)
    
    gmm.train(DTR_pca, LTR, DTE_pca, LTE, None)
    gmm.calculate_scores()
    
    scores = gmm.scores
    
    return scores, LTE


def calibrated_GMM_test_dcf(DTR, LTR, DTE, LTE, prior):
    
    llr, Label = GMM_candidate_test(DTR, LTR, DTE, LTE)
    
    llr_cal, Label_cal = calibrate(llr, Label, 0.5)
    predicted_labels = optimal_bayes_decision(llr_cal, prior, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = MIN_DCF(prior, 1, 1, Label_cal, llr_cal)
    act_dcf = DCF(prior, conf_matrix, 1, 1, True)
    
    print(f"GMM (Test) {prior}     min_dcf: {round(min_dcf, 3)}        act_dcf: {round(act_dcf, 3)}")