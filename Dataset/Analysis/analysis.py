from Utils.load import *
import numpy as np


# Loading Dataset

training_data, training_label = load('./Dataset/Train.txt')
test_data, test_label = load('./Dataset/Test.txt')

# Number of Males and Females of training and test sets

num_male_train = numpy.count_nonzero(training_label == 0)
num_female_train = numpy.count_nonzero(training_label == 1)
num_male_test = numpy.count_nonzero(test_label == 0)
num_female_test = numpy.count_nonzero(test_label == 1)

print(
    "        TRAIN     TESTanalysis\n" +
    f"MALE     {num_male_train}      {num_male_test}\n" +
    f"FEMALE   {num_female_train}     {num_female_test}\n ")

