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
    
    #8 components --> 3 iterations
    gmm = GMM(3,"Tied")
    
    gmm.train(DTR, LTR, DTE, LTE, None)
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
    
    
def TiedGMM_EVAL(DTR, LTR, DTE, LTE, prior):
    
    min_dcf_results = []
    min_dcf_results_znorm = []
    min_dcf_results_eval = []
    min_dcf_results_znorm_eval = []
    
    DTR_znorm = znorm(DTR)
    DTE_znorm = znorm(DTE)
    
    tied = GMM(i,"Tied")
    
    for i in range(5):
        
        SPost, Label = kfold( DTR, LTR, tied, 5, prior)
        res_val = MIN_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_results.append(res_val)
        
        tied.train(DTR, LTR, DTE, LTE, 0.5)
        tied.calculate_scores()
        scores = tied.scores
        res_eval = MIN_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_results_eval.append(res_eval)
        
        SPost, Label = kfold( DTR_znorm, LTR, tied, 5, prior)
        res_val_znorm = MIN_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_results_znorm.append(res_val_znorm)
        
        tied.train(DTR_znorm, LTR, DTE_znorm, LTE, 0.5)
        tied.calculate_scores()
        scores = tied.scores
        res_eval_znorm = MIN_DCF(0.5, 1, 1, LTE, scores)
        min_dcf_results_znorm_eval.append(res_eval_znorm)
        
        print(f"Fold {i} done")
        
    plt.figure()
    
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    
    plt.title("Tied GMM and Tied GMM + Znorm")
    
    plt.bar( x_axis+0.25, min_dcf_results, label="Training", color="orange", alpha=0.5, width=0.25, edgecolor="black", linewidth=1)
    plt.bar( x_axis+0.25, min_dcf_results_eval, label="Evaluation", color="blue", alpha=0.5, width=0.25, edgecolor="black", linewidth=1)
    plt.bar( x_axis+0.25, min_dcf_results_znorm, label="Training Z-Norm", color="green", alpha=0.5, width=0.25, edgecolor="black", linewidth=1)
    plt.bar( x_axis+0.25, min_dcf_results_znorm_eval, label="Evaluation Z-Norm", color="red", alpha=0.5, width=0.25, edgecolor="black", linewidth=1)
    
    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()
    plt.title("Tied GMM")
    plt.xlabel("Iteration")
    plt.ylabel("Minimum DCF")
    plt.savefig("Testing/TiedGMM_EVAL.png")
    plt.close()
    
    
        
        
        