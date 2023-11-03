from Utils.load import *
from Utils.utils import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr


# Loading Dataset

training_data, training_label = load('./Dataset/Train.txt')
test_data, test_label = load('./Dataset/Test.txt')


def plot_heatmap(data, label, color, filename):

    print(f"data prima di essere tagliato: {data.shape}")

    data = data[:, label]

    print(f"data dopo essere tagliato: {data.shape}")

    n_features = data.shape[0]
    n_samples = data.shape[1]

    heatmap = numpy.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if j<=i:
                heatmap[i][j] = abs(pearsonr(data[i, :], data[j, :])[0])
                heatmap[j][i] = heatmap[i][j]
    
    plt.figure()
    plt.xticks(numpy.arange(n_features), numpy.arange(1, n_features + 1))
    
    plt.title("Heatmap of Pearson Correlation")
    plt.imshow(heatmap, cmap=color)
    plt.colorbar()
    
    plt.savefig(filename)
    plt.close()


#Plot heatmap for all the data
plot_heatmap(training_data, range(
    training_data.shape[1]), "YlGn", "Dataset/Analysis/Heatmaps/correlation_train.png")

#Plot the heatmap only for male
plot_heatmap(training_data, training_label==0, "PuBu", "Dataset/Analysis/Heatmaps/male_correlation_train.png")

#Plot the heatmap only for female
plot_heatmap(training_data, training_label==1, "YlOrRd", "Dataset/Analysis/Heatmaps/female_correlation_train.png")