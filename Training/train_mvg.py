from Functions.load import *
from Functions.reshape_functions import *
from Functions.kfold import *
from Models.MVG import *
from Preprocessing.PCA import *
from Metrics.DCF import *
from Metrics.ROC import *


def train_classifier(classifier_list, name, D, L, prior):
    m_list = [11, 10, 9, 8, 7]
    
    print("#####################################\n")
    print(f"{name} NO PCA\n")
    for classifier, prior in classifier_list:
        SPost, Label = kfold(D, L, classifier, 5, prior)
        res = MIN_DCF(prior, 1, 1, Label, SPost)
        print(f"Min DCF ({classifier.name}, {prior}): {round(res, 3)}")
    
    print(f"{name} + PCA(7-8-9-10-11)\n")
    for m in m_list:
        D_pca, _ = PCA(D, m)
        print(f"Value of m: {m}")
        for classifier, prior in classifier_list:
            SPost, Label = kfold(D_pca, L, classifier, 5, prior)
            res = MIN_DCF(prior, 1, 1, Label, SPost)
            print(f"Min DCF ({classifier.name}, {prior}): {round(res, 3)}")
    
    print("\n")
    print("#####################################\n")

def train_LogGaussian_Classifier(D, L):
    LogGaussian_list = [
        (LogGaussian_Classifier(), 0.5),
        (LogGaussian_Classifier(), 0.1),
        (LogGaussian_Classifier(), 0.9),
    ]
    train_classifier(LogGaussian_list, "MVG", D, L)

def train_NBGaussian_Classifier(D, L):
    NBGaussian_list = [
        (NBGaussian_Classifier(), 0.5),
        (NBGaussian_Classifier(), 0.1),
        (NBGaussian_Classifier(), 0.9), 
    ]
    train_classifier(NBGaussian_list, "NB Gaussian", D, L)

def train_TiedGaussian_Classifier(D, L):
    TiedGaussian_list = [
        (TiedGaussian_Classifier(), 0.5),
        (TiedGaussian_Classifier(), 0.1),
        (TiedGaussian_Classifier(), 0.9)
    ]
    train_classifier(TiedGaussian_list, "Tied Gaussian", D, L)

def train_TiedNBGaussian_Classifier(D, L):
    TiedNBGaussian_list = [
        (TiedNBGaussian_Classifier(), 0.5),
        (TiedNBGaussian_Classifier(), 0.1),
        (TiedNBGaussian_Classifier(), 0.9)
    ]
    train_classifier(TiedNBGaussian_list, "TiedNB Gaussian", D, L)