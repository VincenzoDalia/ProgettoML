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
        for j in range(i + 1):
            coef = abs(pearsonr(data[i, :], data[j, :])[0])
            heatmap[i][j] = coef
            heatmap[j][i] = coef

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, cmap=color)

    ax.set_xticks(numpy.arange(n_features))
    ax.set_yticks(numpy.arange(n_features))
    ax.set_xticklabels(numpy.arange(1, n_features + 1))
    ax.set_yticklabels(numpy.arange(1, n_features + 1))

    ax.set_title("Heatmap of Pearson Correlation")
    fig.colorbar(im)

    plt.savefig(filename)
    plt.close(fig)


def plot_heatmaps_dataset(D):
    color = "inferno"
    filename = "Dataset/Analysis/Heatmaps/correlation_all.png"
    plot_heatmap(D, range(D.shape[1]), color, filename)


plot_heatmap(training_data, range(
    training_data.shape[1]), "inferno", "Dataset/Analysis/Heatmaps/correlation_train.png")
# plot_heatmaps_dataset(training_data)

plot_heatmap(training_data, range(
    training_data.shape[1]), "inferno", "Dataset/Analysis/Heatmaps/correlation_train.png")
