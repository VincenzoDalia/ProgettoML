from Functions.load import *
from Functions.reshape_functions import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr



# ------------------- FEATURE ANALYSIS ------------------- #

def plot_histogram(data, labels):
    
    mean_values = np.mean(data, axis=1, keepdims=True)

    
    if np.any(mean_values != 0):
        normalized_data = data - mean_values
    else:
        normalized_data = data

    
    male_data = normalized_data[:, labels == 0]
    female_data = normalized_data[:, labels == 1]

    num_features = data.shape[0] 

    for feature_index in range(num_features):
        
        plt.figure()
        
        
        feature_name = "Feature " + str(feature_index + 1)
        title = "Probability Density - " + feature_name
        plt.title(title, fontsize=10, fontweight="bold")

        
        plt.hist(
            male_data[feature_index, :],
            bins=100,
            density=True,
            alpha=0.5,
            label="Male",
            linewidth=0.3,
            edgecolor="black",
            color='black'
        )
        plt.hist(
            female_data[feature_index, :],
            bins=100,
            density=True,
            alpha=0.5,
            label="Female",
            linewidth=0.3,
            edgecolor="black",
            color="red",
        )


        plt.legend()
        plt.tight_layout()
        plt.savefig("Dataset/Analysis/Histograms/histogram{}.pdf".format(feature_index))


def plot_scatter_matrix(data, labels):
    
    male_data = data[:, labels == 0]
    female_data = data[:, labels == 1]

    num_features = data.shape[0]  

    
    plt.figure()

    
    for feature_index1 in range(num_features):
        for feature_index2 in range(feature_index1 - 1, -1, -1):
            feature_name1 = f"Feature {feature_index1 + 1}"
            feature_name2 = f"Feature {feature_index2 + 1}"

            plt.xlabel(feature_name1)
            plt.ylabel(feature_name2)

            plt.scatter(male_data[feature_index1, :], male_data[feature_index2, :],
                        label="Male", color="blue", alpha=0.5)
            plt.scatter(female_data[feature_index1, :], female_data[feature_index2, :],
                        label="Female", color="red", alpha=0.5)
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"Dataset/Analysis/Scatter/Plot_scatter{feature_index1}_{feature_index2}.pdf")

            
            plt.clf()

    
    plt.close()


def plot_LDA_hist(dataset, label, m):
    
    W1 = LDA1(dataset, label, m)
    
    y1 = numpy.dot(W1.T, dataset)

    dataMale = y1[:, label == 0]
    dataFemale = y1[:, label == 1]

    plt.figure()
    plt.xlabel("LDA Direction")
    plt.hist(
        dataMale[0],
        bins=100,
        density=True,
        alpha=0.8,
        label="Male",
        edgecolor="black",
        color="black")
    plt.hist(
        dataFemale[0],
        bins=100,
        density=True,
        alpha=0.8,
        label="Female",
        edgecolor="black",
        color="red",
    )
    plt.legend()
    plt.savefig("Dataset/Analysis/LDA/lda.pdf")
        
        
def plot_heatmap(data, label, title, color, filename):

    data = data[:, label]

    n_features = data.shape[0]

    heatmap = numpy.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if j<=i:
                heatmap[i][j] = abs(pearsonr(data[i, :], data[j, :])[0])
                heatmap[j][i] = heatmap[i][j]
    
    plt.figure()
    plt.xticks(numpy.arange(n_features), numpy.arange(1, n_features + 1))
    
    plt.title("Heatmap of Pearson Correlation" + title)
    plt.imshow(heatmap, cmap=color)
    plt.colorbar()
    
    plt.savefig(filename)
    plt.close()
    
    
def explained_variance(data):
    
    mu = np.mean(data, axis=1, keepdims=True)
    data_centered = data - mu
    
    cov = np.cov(data_centered)
    
    s = np.linalg.eigvalsh(cov)[::-1]

    explained_variance = s / np.sum(s)

    plt.figure()
    plt.yticks(numpy.linspace(plt.ylim()[0], plt.ylim()[1], 35)) 
    plt.xticks(numpy.linspace(0,12, num=13 ))
    plt.xlim(0,11)
    plt.plot(np.cumsum(explained_variance))
    plt.grid(True)
    plt.xlabel("Components")
    plt.ylabel("Fraction of explained variance")
    plt.savefig("Dataset/Analysis/ExplainedVariance/explained_variance.pdf")


