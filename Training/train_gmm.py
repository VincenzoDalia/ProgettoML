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

### GMM Graphs ##

def calculate_min_dcf_values(D, L,model, ZNorm=False):
    
    min_dcf_values = []
    if ZNorm:
        print("Znorm")
        D = znorm(D)
    for i in range(5):
        
        gmm = GMM(i,model) # Possible models: GMM, Tied, Diagonal, Tied-Diagonal
        
        
        SPost, Label = kfold( D, L, gmm, 5, None)
        res = MIN_DCF(0.5, 1, 1, Label, SPost)
        min_dcf_values.append(res)
        
        print(f"Iteration {i+1} done!")
    return min_dcf_values

def GMM_ZNorm_plot_diff_component(D, L):
    
    raw_values = calculate_min_dcf_values(D, L,"GMM")
    znorm_values= calculate_min_dcf_values(D, L,"GMM", True)

    
    plt.figure()
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    plt.title("GMM and GMM + Znorm")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="Znorm",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/GMM_RAW+znorm.pdf")


def GMM_PCA11_plot_diff_component(D, L):
    
    D_pca, _ = PCA(D, 11)
    raw_values = calculate_min_dcf_values(D, L, "GMM")
    znorm_values= calculate_min_dcf_values(D_pca, L, "GMM",True)


    plt.figure()
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    plt.title("GMM and GMM + PCA 11")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="PCA",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/GMM_RAW+PCA11.pdf")


def GMM_PCA10_plot_diff_component(D, L):
    
    D_pca, _ = PCA(D, 10)
    raw_values = calculate_min_dcf_values(D, L, "GMM")
    znorm_values= calculate_min_dcf_values(D_pca, L, "GMM", True)


    plt.figure()
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    plt.title("GMM and GMM + PCA 10")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="PCA",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/GMM_RAW+PCA10.pdf")


def TiedGMM_ZNorm_plot_diff_component(D, L):
    
    raw_values = calculate_min_dcf_values(D, L,"Tied")
    znorm_values= calculate_min_dcf_values(D, L,"Tied", True)


    plt.figure()
    plt.xlabel("TiedGMM components")
    plt.ylabel("minDCF")
    plt.title("TiedGMM and TiedGMM + Znorm")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="Znorm",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/TiedGMM_RAW+znorm.pdf")


def TiedGMM_PCA11_plot_diff_component(D, L):
    
    D_pca, _ = PCA(D, 11)
    raw_values = calculate_min_dcf_values(D, L, "Tied")
    znorm_values= calculate_min_dcf_values(D_pca, L, "Tied", True)


    plt.figure()
    plt.xlabel("TiedGMM components")
    plt.ylabel("minDCF")
    plt.title("TiedGMM and TiedGMM + PCA 11")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="PCA",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/TiedGMM_RAW+PCA11.pdf")


def TiedGMM_PCA10_plot_diff_component(D, L):
    
    D_pca, _ = PCA(D, 10)
    raw_values = calculate_min_dcf_values(D, L, "Tied")
    znorm_values= calculate_min_dcf_values(D_pca, L, "Tied", True)


    plt.figure()
    plt.xlabel("TiedGMM components")
    plt.ylabel("minDCF")
    plt.title("TiedGMM and TiedGMM + PCA 10")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="PCA",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/TiedGMM_RAW+PCA10.pdf")


def DiagonalGMM_ZNorm_plot_diff_component(D, L):
    
    raw_values = calculate_min_dcf_values(D, L,"Diagonal")
    znorm_values= calculate_min_dcf_values(D, L,"Diagonal", True)


    plt.figure()
    plt.xlabel("DiagonalGMM components")
    plt.ylabel("minDCF")
    plt.title("DiagonalGMM and DiagonalGMM + Znorm")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="Znorm",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/DiagonalGMM_RAW+znorm.pdf")


def DiagonalGMM_PCA11_plot_diff_component(D, L):
    
    D_pca, _ = PCA(D, 11)
    raw_values = calculate_min_dcf_values(D, L, "Diagonal")
    znorm_values= calculate_min_dcf_values(D_pca, L, "Diagonal", True)


    plt.figure()
    plt.xlabel("DiagonalGMM components")
    plt.ylabel("minDCF")
    plt.title("DiagonalGMM and DiagonalGMM + PCA 11")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="PCA",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/DiagonalGMM_RAW+PCA11.pdf")


def DiagonalGMM_PCA10_plot_diff_component(D, L):
    
    D_pca, _ = PCA(D, 10)
    raw_values = calculate_min_dcf_values(D, L, "Diagonal")
    znorm_values= calculate_min_dcf_values(D_pca, L, "Diagonal", True)


    plt.figure()
    plt.xlabel("DiagonalGMM components")
    plt.ylabel("minDCF")
    plt.title("DiagonalGMM and DiagonalGMM + PCA 10")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="PCA",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/DiagonalGMM_RAW+PCA10.pdf")


def TiedDiagonalGMM_ZNorm_plot_diff_component(D, L):
    
    raw_values = calculate_min_dcf_values(D, L,"Tied-Diagonal")
    znorm_values= calculate_min_dcf_values(D, L,"Tied-Diagonal", True)


    plt.figure()
    plt.xlabel("Tied-DiagonalGMM components")
    plt.ylabel("minDCF")
    plt.title("Tied-DiagonalGMM and Tied-DiagonalGMM + Znorm")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="Znorm",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/TiedDiagonalGMM_RAW+znorm.pdf")


def TiedDiagonalGMM_PCA11_plot_diff_component(D, L):
    
    D_pca, _ = PCA(D, 11)
    raw_values = calculate_min_dcf_values(D, L, "Tied-Diagonal")
    znorm_values= calculate_min_dcf_values(D_pca, L, "Tied-Diagonal", True)


    plt.figure()
    plt.xlabel("Tied-DiagonalGMM components")
    plt.ylabel("minDCF")
    plt.title("Tied-DiagonalGMM and Tied-DiagonalGMM + PCA 11")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="PCA",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/TiedDiagonalGMM_RAW+PCA11.pdf")


def TiedDiagonalGMM_PCA10_plot_diff_component(D, L):
    
    D_pca, _ = PCA(D, 10)
    raw_values = calculate_min_dcf_values(D, L, "Tied-Diagonal")
    znorm_values= calculate_min_dcf_values(D_pca, L, "Tied-Diagonal", True)


    plt.figure()
    plt.xlabel("Tied-DiagonalGMM components")
    plt.ylabel("minDCF")
    plt.title("Tied-DiagonalGMM and Tied-DiagonalGMM + PCA 10")
    iterations = range(5)
    x_axis = np.arange(len(iterations))
    bounds = np.array(iterations)
    plt.bar(
        x_axis + 0.00,
        raw_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Orange",
        label="RAW",
    )
    plt.bar(
        x_axis + 0.25,
        znorm_values,
        width=0.25,
        linewidth=1.0,
        edgecolor="black",
        color="Green",
        label="PCA",
    )

    plt.xticks([r + 0.125 for r in range(len(bounds))], [2**i for i in bounds])
    plt.legend()

    plt.savefig("Training/GMM_Plot/TiedDiagonalGMM_RAW+PCA10.pdf")


def GMM_candidate(D,L):
    
    #4 components --> 2 iterations
    gmm = GMM(2,"GMM")
    
    D_pca, _ = PCA(D, 11)
    SPost, Label = kfold(D_pca, L, gmm, 5, None)
    
    return SPost, Label


def calibrated_GMM_dcf(D, L, prior):
    print(f"GMM - min_dcf / act_dcf  {prior} \n")
    llr, Label = GMM_candidate(D, L)
    llr_cal, Label_cal = calibrate(llr, Label, 0.5)
    predicted_labels = optimal_bayes_decision(llr_cal, prior, 1, 1)
    conf_matrix = confusionMatrix(Label_cal, predicted_labels)
    min_dcf = MIN_DCF(prior, 1, 1, Label_cal, llr_cal)
    act_dcf = DCF(prior, conf_matrix, 1, 1, True)
    print("GMM (train) min_dcf: ", round(min_dcf, 3))
    print("GMM (train) act_dcf: ", round(act_dcf, 3))


 
### GMM Tabelle ##
def run_gmm_model(gmm, components, pi, message, D, L):
    SPost, Label = kfold(D, L, gmm, 5, None)
    res = MIN_DCF(pi, 1, 1, Label, SPost)
    print(message, "pi = ", pi, str(2**components) + " components: ", round(res, 3))

def GMM_diff_priors(D, L):
    for i in range(1, 3):
        for pi in [0.5, 0.1, 0.9]:
            run_gmm_model(GMM(i,"GMM"), i, pi, "GMM min_DCF", D, L)

    for pi in [0.5, 0.1, 0.9]:
        run_gmm_model(GMM(3, "Tied"), 3, pi, "Tied_GMM min_DCF", D, L)
 
    D_pca, _ = PCA(D, 11)

    for pi in [0.5, 0.1, 0.9]:
        run_gmm_model(GMM(2,"GMM"), 2, pi, "GMM min_DCF + PCA(11)", D_pca, L)
        
    D_pca, _ = PCA(D, 10)

    for pi in [0.5, 0.1, 0.9]:
        run_gmm_model(GMM(2,"GMM"), 2, pi, "GMM min_DCF + PCA(10)", D_pca, L)
        
    D_pca, _ = PCA(D, 9)

    for pi in [0.5, 0.1, 0.9]:
        run_gmm_model(GMM(2,"GMM"), 2, pi, "GMM min_DCF + PCA(9)", D_pca, L)
        
        
def GMM_diff_priors_znorm(D, L):
    D = znorm(D)
    for i in range(1, 3):
        for pi in [0.5, 0.1, 0.9]:
            run_gmm_model(GMM(i,"GMM"), i, pi, "GMM min_DCF + ZNorm",D, L)

    for pi in [0.5, 0.1, 0.9]:
        run_gmm_model(GMM(3,"Tied"), 3, pi, "GMM Tied min_DCF + ZNorm", D, L)