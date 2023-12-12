from Functions.load import *
from Dataset.Analysis.analysis import *
import numpy as np
from Models.MVG import *
from Training.train_mvg import *
#from Training.train_logistic_regression import *
from Training.train_logistic_regression import *
from Training.train_svm import *
from Training.train_gmm import *
from Calibration.Calibrate import *
from Metrics.BayesErr import * 
from Metrics.ROC import * 
from Testing.test_gmm import *
from Testing.test_lr import *
from Testing.test_svm import *




if __name__ == '__main__':
    # Loading Dataset

    training_data, training_label = load('./Dataset/Train.txt')
    test_data, test_label = load('./Dataset/Test.txt')

    num_features, tot_train = np.shape(training_data)
    tot_test = np.shape(test_data)[1]

    print(f"The number of features is {num_features}")

    print(
        f"\nThe total number of samples for the training set is: {tot_train}")
    print(
        f"The total number of samples for the test set is: {tot_test}\n\n")

    # Number of Males and Females of training and test sets

    num_male_train = np.count_nonzero(training_label == 0)
    num_female_train = np.count_nonzero(training_label == 1)
    num_male_test = np.count_nonzero(test_label == 0)
    num_female_test = np.count_nonzero(test_label == 1)

    tot_male = num_male_train + num_male_test
    tot_female = num_female_train + num_female_test

    print(
        f"There is a {num_male_train/tot_train * 100}% of male over the training set")
    print(
        f"There is a {num_female_train/tot_train * 100}% of female over the training set\n")
    print(
        f"There is a {num_male_test/tot_test * 100}% of male over the test set")
    print(
        f"There is a {num_female_test/tot_test * 100}% of female over the test set\n")

    print(
        "        TRAIN     TESTmain\n" +
        f"MALE     {num_male_train}      {num_male_test}\n" +
        f"FEMALE   {num_female_train}     {num_female_test}\n ")
    


    # Train Gaussian Classifiers #
    
    #print("Training Gaussian Classifiers...\n")
    #train_LogGaussian_Classifier(training_data, training_label)
    #train_NBGaussian_Classifier(training_data, training_label)
    #train_TiedGaussian_Classifier(training_data, training_label)
    #train_TiedNBGaussian_Classifier(training_data, training_label)
    #print("Training Gaussian Classifiers... Done\n")
        
        
    # Train Logistic Regression #
    
    #print("Training Logistic Regression...\n")
    
    #comparation_plot(training_data, training_label, 0.5)
    #comparation_plot_2(training_data, training_label, 0.5)
    #simple_Logistic_Regression_Graph(training_data,training_label,0.5)
    #ZNorm_Logistic_Regression_Graph(training_data, training_label, 0.5)
    #different_Pi_T_Raw_Graph(training_data, training_label)
    #different_Pi_T_znorm_Graph(training_data, training_label)
    #PCA_Logistic_Regression_Graph_various_Pi(training_data, training_label, 0.5)
    #PCA_Logistic_Regression_Graph_various_Pi(training_data, training_label, 0.9)
    #PCA_Logistic_Regression_Graph_various_Pi(training_data, training_label, 0.1)
    #PCA_Logistic_Regression_Graph_various_Pi_T(training_data, training_label, 0.5)
    #PCA_Logistic_Regression_Graph_various_Pi_T(training_data, training_label, 0.9)
    #PCA_Logistic_Regression_Graph_various_Pi_T(training_data, training_label, 0.1)
    
    #LR_diff_priors(training_data,training_label)
    #LR_diff_priors_zscore(training_data,training_label)
    
    # ------------------ Quadratic Logistic Regression Training ------------------ #
    
    #quadratic_comparation_plot(training_data, training_label, 0.5)
    #quadratic_comparation_plot_2(training_data, training_label, 0.5)
    
    #Quad_LR_diff_priors(training_data,training_label)
    #Quad_LR_diff_priors(training_data,training_label, True)
    
    #print("Training Logistic Regression... Done\n")

    # Train SVM #
    
    #print("Training SVM...\n")
    
    #svm_comparation_plot(training_data, training_label, 0.5, 1)
    #print("SVM comparation plot done\n")
    #linear_svm_raw_plot(training_data, training_label, 0.5, 1)
    #print("Linear SVM raw plot done\n")
    #linear_svm_znorm_plot(training_data, training_label, 0.5, 1)
    #print("Linear SVM znorm plot done\n")
    #polynomial_svm_raw_plot(training_data, training_label, 0.5, 1)
    #print("Polynomial SVM raw plot done\n")
    #polynomial_svm_znorm_plot(training_data, training_label, 0.5, 1)
    #print("Polynomial SVM znorm plot done\n")
    #radial_svm_raw_plot(training_data, training_label, 0.5, 1)
    #print("Radial SVM raw plot done\n")
    #radial_svm_znorm_plot(training_data, training_label, 0.5, 1)
    #print("Radial SVM znorm plot done\n")

    #SVM_diff_priors(training_data, training_label)
    #SVM_diff_priors(training_data, training_label, True) # ZNorm
    #RadKernBased_diff_priors(training_data, training_label)
    #RadKernBased_diff_priors(training_data, training_label, True)  # ZNorm
    #Poly_SVM_diff_priors(training_data, training_label)
    #Poly_SVM_diff_priors(training_data, training_label, True)  # ZNorm
    
    #print("Training SVM... Done\n")
    
    
    # Train GMM #
    
     ### Graphs ###
    """  
    print("Training GMM...\n")
    print("Plotting GMM RAW and ZNorm...")
    #GMM_ZNorm_plot_diff_component(training_data, training_label)
    print("Plotting GMM RAW and ZNorm... Done\n")
    print("Plotting GMM RAW and PCA11...") 
    GMM_PCA11_plot_diff_component(training_data, training_label)
    print("Plotting GMM RAW and PCA11... Done\n")
    print("Plotting GMM RAW and PCA10...")
    GMM_PCA10_plot_diff_component(training_data, training_label)
    print("Plotting GMM RAW and PCA10... Done\n")
    print("Plotting TiedGMM RAW and ZNorm...")
    TiedGMM_ZNorm_plot_diff_component(training_data, training_label)
    print("Plotting TiedGMM RAW and ZNorm... Done\n")
    print("Plotting TiedGMM RAW and PCA11...")
    TiedGMM_PCA11_plot_diff_component(training_data, training_label)
    print("Plotting TiedGMM RAW and PCA11... Done\n")
    print("Plotting TiedGMM RAW and PCA10...")
    TiedGMM_PCA10_plot_diff_component(training_data, training_label)
    print("Plotting TiedGMM RAW and PCA10... Done\n")
    print("Plotting DiagGMM RAW and ZNorm...")
    DiagonalGMM_ZNorm_plot_diff_component(training_data, training_label)
    print("Plotting DiagGMM RAW and ZNorm... Done\n")
    print("Plotting DiagGMM RAW and PCA11...")
    DiagonalGMM_PCA11_plot_diff_component(training_data, training_label)
    print("Plotting DiagGMM RAW and PCA11... Done\n")
    print("Plotting DiagGMM RAW and PCA10...")
    DiagonalGMM_PCA10_plot_diff_component(training_data, training_label)
    print("Plotting DiagGMM RAW and PCA10... Done\n")
    print("Plotting TiedDiagGMM RAW and ZNorm...")
    TiedDiagonalGMM_ZNorm_plot_diff_component(training_data, training_label)
    print("Plotting TiedDiagGMM RAW and ZNorm... Done\n")
    print("Plotting TiedDiagGMM RAW and PCA11...")
    TiedDiagonalGMM_PCA11_plot_diff_component(training_data, training_label)
    print("Plotting TiedDiagGMM RAW and PCA11... Done\n")
    print("Plotting TiedDiagGMM RAW and PCA10...")
    TiedDiagonalGMM_PCA10_plot_diff_component(training_data, training_label)
    print("Plotting TiedDiagGMM RAW and PCA10... Done\n")
    print("GMM Graphs... Done\n") """
    
    #GMM_diff_priors(training_data, training_label)
    #GMM_diff_priors_znorm(training_data, training_label)
    
    
    
        # ----------------- Calibration ----------------- #
    
    #Plot un-calibrated candidate models
    """ scores,labels = LR_candidate_train(training_data, training_label)
    Bayes_Error(labels, scores, "Uncalibrated_LR")
    print("Uncalibrated LR... Done\n")
    scores,labels = SVM_candidate_train(training_data, training_label)
    Bayes_Error(labels, scores, "Uncalibrated_SVM")
    print("Uncalibrated SVM... Done\n")
    scores,labels = GMM_candidate_train(training_data, training_label)
    Bayes_Error(labels, scores, "Uncalibrated_GMM")
    print("Uncalibrated GMM... Done\n") 
    """
    
    #Plot ucalibrated candidate models
    """
    scores,labels = LR_candidate_train(training_data, training_label)
    calibrated_scores, calibrated_labels = calibrate(scores, labels,0.5)
    Bayes_Error(calibrated_labels, calibrated_scores, "Calibrated_LR")
    print("Calibrated LR... Done\n")
    
    scores,labels = SVM_candidate_train(training_data, training_label)
    calibrated_scores, calibrated_labels = calibrate(scores, labels,0.5)
    Bayes_Error(calibrated_labels, calibrated_scores, "Calibrated_SVM")
    print("Calibrated SVM... Done\n")
    
    scores,labels = GMM_candidate_train(training_data, training_label)
    calibrated_scores, calibrated_labels = calibrate(scores, labels,0.5)
    Bayes_Error(calibrated_labels, calibrated_scores, "Calibrated_GMM")
    print("Calibrated GMM... Done\n")  """
    
    
    # ---------------- Validation and Evaluation ---------------- #
    
    
    """ print("\n")
    print("Computing minDCF and actDCF for Validation and Evaluation of Calibrated Models...\n")
    
    # pi = 0.5
    calibrated_LR_train_dcf(training_data, training_label, 0.5)
    calibrated_SVM_train_dcf(training_data, training_label, 0.5)
    calibrated_GMM_train_dcf(training_data, training_label, 0.5)
    
    calibrated_LR_test_dcf(training_data, training_label, test_data, test_label, 0.5)
    calibrated_SVM_test_dcf(training_data, training_label, test_data, test_label, 0.5)
    calibrated_GMM_test_dcf(training_data, training_label, test_data, test_label, 0.5)
    
    print("\n")    
    
    # pi = 0.1
    calibrated_LR_train_dcf(training_data, training_label, 0.1)
    calibrated_SVM_train_dcf(training_data, training_label, 0.1)
    calibrated_GMM_train_dcf(training_data, training_label, 0.1)
    
    calibrated_LR_test_dcf(training_data, training_label, test_data, test_label, 0.1)
    calibrated_SVM_test_dcf(training_data, training_label, test_data, test_label, 0.1)
    calibrated_GMM_test_dcf(training_data, training_label, test_data, test_label, 0.1) 
    
    print("\n")
    
    # pi = 0.9
    calibrated_LR_train_dcf(training_data, training_label, 0.9)
    calibrated_SVM_train_dcf(training_data, training_label, 0.9)
    calibrated_GMM_train_dcf(training_data, training_label, 0.9)
    
    calibrated_LR_test_dcf(training_data, training_label, test_data, test_label, 0.9)
    calibrated_SVM_test_dcf(training_data, training_label, test_data, test_label, 0.9)
    calibrated_GMM_test_dcf(training_data, training_label, test_data, test_label, 0.9) 
    
    print("Computing minDCF and actDCF for Validation and Evaluation of Calibrated Models... Done\n")
    print("\n") """
    
    
    ## ROC e Bayes Error Comparison between models (Evaluation)##
    
    print("Computing scores and labels for LR, SVM and GMM ...")
    LR_scores, LR_labels = LR_candidate_test(training_data, training_label, test_data, test_label)
    SVM_scores, SVM_labels = SVM_candidate_test(training_data, training_label, test_data, test_label)
    GMM_scores, GMM_labels = GMM_candidate_test(training_data, training_label, test_data, test_label)
    
    print("Computing scores and labels for LR, SVM and GMM ...Done!")
    
    print("Computing calibrated scores and labels for LR, SVM and GMM ...")
    LR_calibrated_scores, LR_calibrated_labels = calibrate(LR_scores, LR_labels,0.5)
    SVM_calibrated_scores, SVM_calibrated_labels = calibrate(SVM_scores, SVM_labels,0.5)
    GMM_calibrated_scores, GMM_calibrated_labels = calibrate(GMM_scores, GMM_labels,0.5)
    print("Computing calibrated scores and labels for LR, SVM and GMM ...Done!\n")
    
    print("Plotting ROC curves for Calibrated Models (Evaluation)...\n")
    plot_ROC_comparison(LR_calibrated_scores, LR_calibrated_labels, SVM_calibrated_scores, SVM_calibrated_labels, GMM_calibrated_scores, GMM_calibrated_labels)
    print("Plotting ROC curves for Calibrated Models (Evaluation)... Done\n")
    
    print("Plotting Bayes Error for Calibrated Models (Evaluation)...\n")
    print("Plotting Bayes Error for Calibrated Models (Evaluation)... Done\n")