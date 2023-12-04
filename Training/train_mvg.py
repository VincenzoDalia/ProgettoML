from Functions.load import *
from Functions.reshape_functions import *
from Functions.kfold import *
from Models.MVG import *
from Preprocessing.PCA import *


def train_MVG(D,L):
    MVG_list = [
        (LogGaussianClassifier(), 0.5),
        (LogGaussianClassifier(), 0.1),
        (LogGaussianClassifier(), 0.9),
    ]
    
    m_list = [11, 10, 9, 8, 7]
    print("#####################################\n")
    
    print("MVG NO PCA\n")
    for classifier, value in MVG_list:
        SPost, Label = kfold(classifier, 5, D, L, value)
        res = min_DCF(value, 1, 1, Label, SPost)
        print(f"Min DCF ({classifier.name}, {value}): {round(res, 3)}")
    
    
    print("MVG + PCA(7-8-9-10-11)\n")
    for m in m_list:
        D_pca,_ = PCA(D, m)
        print(f"Value of m: {m}")   
        for classifier, value in MVG_list:
            SPost, Label = kfold(classifier, 5, D_pca, L, value)
            res = min_DCF(value, 1, 1, Label, SPost)
            print(f"Min DCF , {value}): {round(res, 3)}")
            
    print("\n") 
    print("#####################################\n")
         

def train_NB(D,L):
    NB_list = [
        
        (NaiveBayesGaussianClassifier(), 0.5),
        (NaiveBayesGaussianClassifier(), 0.1),
        (NaiveBayesGaussianClassifier(), 0.9), 
    ]
    
    m_list = [11, 10, 9, 8, 7]
    print("#####################################\n")
    print("Naive Bayes NO PCA\n")
    for classifier, value in NB_list:
        SPost, Label = kfold(classifier, 5, D, L, value)
        res = min_DCF(value, 1, 1, Label, SPost)
        print(f"Min DCF ({classifier.name}, {value}): {round(res, 3)}")
    
    print("Naive Bayes + PCA(7-8-9-10-11)\n")
    for m in m_list:
        D_pca,_ = PCA(D, m)
        print(f"Value of m: {m}")   
        for classifier, value in NB_list:
            SPost, Label = kfold(classifier, 5, D_pca, L, value)
            res = min_DCF(value, 1, 1, Label, SPost)
            print(f"Min DCF ({classifier.name}, {value}): {round(res, 3)}")
            
    print("\n") 
    print("#####################################\n")
    

def train_TMVG(D,L):
    TMVG_list = [
        
        (TiedGaussianClassifier(), 0.5),
        (TiedGaussianClassifier(), 0.1),
        (TiedGaussianClassifier(), 0.9)
    ]
    
    m_list = [11, 10, 9, 8, 7]
    print("#####################################\n")
    
    print("TMVG NO PCA\n")
    for classifier, value in TMVG_list:
        SPost, Label = kfold(classifier, 5, D, L, value)
        res = min_DCF(value, 1, 1, Label, SPost)
        print(f"Min DCF ({classifier.name}, {value}): {round(res, 3)}")
        
    print("\n")
    
    print("TMVG + PCA(7-8-9-10-11)\n")
    for m in m_list:
        D_pca,_ = PCA(D, m)
        print(f"Value of m: {m}")   
        for classifier, value in TMVG_list:
            SPost, Label = kfold(classifier, 5, D_pca, L, value)
            res = min_DCF(value, 1, 1, Label, SPost)
            print(f"Min DCF ({classifier.name}, {value}): {round(res, 3)}") 
            
    print("\n")
    print("#####################################\n")
            

def train_TNB(D,L):
    TNB_list = [
        
        (TiedNaiveBayesGaussianClassifier(), 0.5),
        (TiedNaiveBayesGaussianClassifier(), 0.1),
        (TiedNaiveBayesGaussianClassifier(), 0.9)
    ]
    
    m_list = [11, 10, 9, 8, 7]
    print("#####################################\n")
    
    print("TNB NO PCA\n")
    for classifier, value in TNB_list:
        SPost, Label = kfold(classifier, 5, D, L, value)
        res = min_DCF(value, 1, 1, Label, SPost)
        print(f"Min DCF ({classifier.name}, {value}): {round(res, 3)}")
        
    print("\n")
    
    print("TNB + PCA(7-8-9-10-11)\n")
    for m in m_list:
        D_pca,_ = PCA(D, m)
        print(f"Value of m: {m}")   
        for classifier, value in TNB_list:
            SPost, Label = kfold(classifier, 5, D_pca, L, value)
            res = min_DCF(value, 1, 1, Label, SPost)
            print(f"Min DCF ({classifier.name}, {value}): {round(res, 3)}") 
              
    print("\n")     
    print("#####################################\n")       
          
