from Functions.load import *
from Functions.reshape_functions import *
from Functions.kfold import *
from Models.LogisticRegression import *
from Preprocessing.PCA import *
from Metrics.DCF import *
from Metrics.ROC import *
from Preprocessing.ZNorm import *
import matplotlib.pyplot as plt
from Metrics.BayesErr import *
from Calibration.Calibrate import *

def LR_candidate_test(DTR,LTR,DTE,LTE):
    
    l = 0.01
    pi_T = 0.1
    
    lr = BinaryLogisticRegression(l)
    
    lr.train(DTR, LTR, DTE, LTE, pi_T)
    lr.calculate_scores()
    
    scores = lr.scores
    
    return scores, LTE


def calibrated_LR_test_dcf(DTR, LTR, DTE, LTE, prior):
    
    llr, Label = LR_candidate_test(DTR, LTR, DTE, LTE)
    
    llr_cal, Label_cal = calibrate(llr, Label, 0.5)
    predicted_labels = optimal_bayes_decision(llr_cal, prior, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = MIN_DCF(prior, 1, 1, Label_cal, llr_cal)
    act_dcf = DCF(prior, conf_matrix, 1, 1, True)
    
    print(f"LR (Test) {prior}     min_dcf: {round(min_dcf, 3)}        act_dcf: {round(act_dcf, 3)}")
    
    
    
def LR_Raw_Eval(DTR, LTR, DTE, LTE, prior, ZNorm = False):
    
    l_values = np.logspace(-5, 5, num=21)
    value = [0.5, 0.1, 0.9]
    name = ""
    
    min_dcf_05 = []
    min_dcf_09 = []
    min_dcf_01 = []
    min_dcf_05_eval = []
    min_dcf_09_eval = []
    min_dcf_01_eval = []
    
    regression = BinaryLogisticRegression(0)
    regression_eval = BinaryLogisticRegression(0)
    
    if ZNorm:
        name = "ZNorm"
        DTR = znorm(DTR)
        DTE = znorm(DTE)

    
    for i, l in enumerate(l_values):
        regression.l = l
        regression_eval.l = l

        SPost, Label = kfold(DTR, LTR, regression, 5, prior)
        min_dcf_05.append(MIN_DCF(value[0], 1, 1, Label, SPost))
        min_dcf_01.append(MIN_DCF(value[1], 1, 1, Label, SPost))
        min_dcf_09.append(MIN_DCF(value[2], 1, 1, Label, SPost))

        regression_eval.train(DTR, LTR, DTE, LTE, 0.5)
        regression_eval.calculate_scores()
        scores = regression_eval.scores
        min_dcf_05_eval.append(MIN_DCF(value[0], 1, 1, LTE, scores))
        min_dcf_01_eval.append(MIN_DCF(value[1], 1, 1, LTE, scores))
        min_dcf_09_eval.append(MIN_DCF(value[2], 1, 1, LTE, scores))
        
        
    plt.figure(figsize=(8,6))
    plt.plot(l_values, min_dcf_05, label="min DCF 0.5", color="b", linestyle="-")
    plt.plot(l_values, min_dcf_01, label="min DCF 0.1", color="r", linestyle="-")
    plt.plot(l_values, min_dcf_09, label="min DCF 0.9", color="g", linestyle="-")
    plt.plot(l_values, min_dcf_05_eval, label="min DCF 0.5 eval", color="b", linestyle="--")
    plt.plot(l_values, min_dcf_01_eval, label="min DCF 0.1 eval", color="r", linestyle="--")
    plt.plot(l_values, min_dcf_09_eval, label="min DCF 0.9 eval", color="g", linestyle="--")
    plt.title(f"Logistic Regression Eval {name}")
    plt.ylim([0, 1.1])
    plt.xlim([0, 1])
    plt.legend()
    plt.savefig(f"Testing/LR_Raw_Eval {name}.png")