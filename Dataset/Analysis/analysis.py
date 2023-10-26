from Utils.load import *
import numpy as np
import matplotlib.pyplot as plt


# Loading Dataset

training_data, training_label = load('./Dataset/Train.txt')
test_data, test_label = load('./Dataset/Test.txt')

num_features, tot_train = np.shape(training_data)
tot_test = np.shape(test_data)[1]


# ------------------- DATASET ANALYSIS ------------------- #

print(f"The number of features is {num_features}")

print(
    f"\nThe total number of samples for the training set is: {tot_train}")
print(
    f"The total number of samples for the test set is: {tot_test}\n\n")

# Number of Males and Females of training and test sets

num_male_train = numpy.count_nonzero(training_label == 0)
num_female_train = numpy.count_nonzero(training_label == 1)
num_male_test = numpy.count_nonzero(test_label == 0)
num_female_test = numpy.count_nonzero(test_label == 1)

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


# ------------------- FEATURE ANALYSIS ------------------- #
def plot_centered_hist(dataset, label):

    D0 = dataset[:, label == 0]
    D1 = dataset[:, label == 1]

    hFea = {
        0: "Feature 1",
        1: "Feature 2",
        2: "Feature 3",
        3: "Feature 4",
        4: "Feature 5",
        5: "Feature 6",
        6: "Feature 7",
        7: "Feature 8",
        8: "Feature 9",
        9: "Feature 10",
        10: "Feature 11",
        11: "Feature 12",
    }

    for dIdx, h in hFea.items():
        
        ax = plt.subplots()[1]

        ax.set_xlabel(h, fontsize=10, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=10, fontweight="bold")

        ax.hist(
            D0[dIdx, :],
            bins=80,
            density=True,
            alpha=0.4,
            label="Male",
            linewidth=0.3,
            edgecolor="black",
        )
        ax.hist(
            D1[dIdx, :],
            bins=80,
            density=True,
            alpha=0.4,
            label="Female",
            linewidth=0.3,
            edgecolor="black",
            color="red",
        )

        ax.legend()

        plt.tight_layout()
        plt.savefig("Dataset/Analysis/Histograms/hist_{}.pdf".format(dIdx))


plot_centered_hist(training_data, training_label)
