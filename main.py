from load import *



if __name__ == '__main__':
      
      
### LOADING DATASET ###
      trainingData, trainingLabels = load('Dataset/Train.txt')
      testData, testLabels = load('Dataset/Test.txt')
      
### DATASET INFO ###

      number_male_tr = numpy.count_nonzero(trainingLabels == 0)
      number_female_tr = numpy.count_nonzero(trainingLabels == 1)
      print("Number of Male in the training set : ", number_male_tr)
      print("Number of female in the training set : ",number_female_tr)