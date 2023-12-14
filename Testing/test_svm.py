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


def calibrated_SVM_test_dcf(DTR, LTR, DTE, LTE, prior):
    
    llr, Label = SVM_candidate_test(DTR, LTR, DTE, LTE)
    
    llr_cal, Label_cal = calibrate(llr, Label, 0.5)
    predicted_labels = optimal_bayes_decision(llr_cal, prior, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = MIN_DCF(prior, 1, 1, Label_cal, llr_cal)
    act_dcf = DCF(prior, conf_matrix, 1, 1, True)
    
    print(f"Radial SVM (Test) {prior}     min_dcf: {round(min_dcf, 3)}        act_dcf: {round(act_dcf, 3)}")
    
    
def RadialSVM_EVAL(DTR, LTR, DTE, LTE, prior, ZNorm=False):
    
    string_to_append = "_raw"
    
    if ZNorm:
        DTR = znorm(DTR)
        DTE = znorm(DTE)
        string_to_append = "_znorm"
    
    C_values = numpy.logspace(-5, 5, num=11)
    gamma_values = [0.1, 0.01, 0.001]
    min_dcf_results = {gamma: {'val': [], 'eval': []} for gamma in gamma_values}

    for gamma in gamma_values:
        for c in C_values:
            svm = Radial_SVM(1, c, gamma)

            SPost, Label = kfold( DTR, LTR, svm, 5, prior)
            res_val = MIN_DCF(0.5, 1, 1, Label, SPost)
            min_dcf_results[gamma]['val'].append(res_val)

            svm.train(DTR, LTR, DTE, LTE, 0.5)
            svm.calculate_scores()
            scores = svm.scores
            res_eval = MIN_DCF(0.5, 1, 1, LTE, scores)
            min_dcf_results[gamma]['eval'].append(res_eval)

    plt.figure()
    plt.xlabel("C")
    plt.xscale("log")
    plt.ylabel("minDCF")
    plt.title("RKB-SVM")

    colors = ['red', 'blue', 'green']
    for gamma, color in zip(gamma_values, colors):
        plt.plot(C_values, min_dcf_results[gamma]['val'], label=f"gamma= {gamma} (VAL)", color=color)
        plt.plot(C_values, min_dcf_results[gamma]['eval'], label=f"gamma = {gamma} (EVAL)", color=color, linestyle="dashed")

    plt.xlim(C_values[0], C_values[-1])
    plt.legend()
    plt.savefig(f"Testing/RadialSVM{string_to_append}.pdf")
    plt.close()