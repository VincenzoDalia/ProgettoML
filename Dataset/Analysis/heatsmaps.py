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


def plot_heatmap(D, L, cmap_name, filename):
    D = D[:, L]
    features = D.shape[0]
    heatmap = numpy.zeros((features, features))

    for i in range(D.shape[0]):
        for j in range(i + 1):
            coef = abs(pearsonr(D[i, :], D[j, :])[0])
            heatmap[i][j] = coef
            heatmap[j][i] = coef

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, cmap=cmap_name)

    ax.set_xticks(numpy.arange(D.shape[0]))
    ax.set_yticks(numpy.arange(D.shape[0]))
    ax.set_xticklabels(numpy.arange(1, D.shape[0] + 1))
    ax.set_yticklabels(numpy.arange(1, D.shape[0] + 1))

    ax.set_title("Heatmap of Pearson Correlation")
    fig.colorbar(im)

    plt.savefig(filename)
    plt.close(fig)


def plot_heatmaps_dataset(D):
    cmap_name = "Greys"
    filename = "./Heatmap/correlation_all.png"
    plot_heatmap(D, range(D.shape[1]), cmap_name, filename)
