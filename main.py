from Utils.load import *
from Dataset.Analysis.analysis import *
import numpy as np


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
    


