from Utils.load import *


if __name__ == '__main__':

    ### LOADING DATASET ###
    trainingData, trainingLabels = load('Dataset/Train.txt')
    testData, testLabels = load('Dataset/Test.txt')

###  DATASET INFO ###

    number_male_train = numpy.count_nonzero(trainingLabels == 0)
    number_female_train = numpy.count_nonzero(trainingLabels == 1)
    number_male_test = numpy.count_nonzero(trainingLabels == 0)
    number_female_test = numpy.count_nonzero(trainingLabels == 1)
    print("Number of Male in the training set : ", number_male_train)
    print("Number of female in the training set : ", number_female_train)
    print("Number of Male in the training set : ", number_male_test)
    print("Number of female in the training set : ", number_female_test)
