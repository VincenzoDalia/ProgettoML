from Utils.load import *
from Utils.utils import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


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
    "        TRAIN     TEST\n" +
    f"MALE     {num_male_train}      {num_male_test}\n" +
    f"FEMALE   {num_female_train}     {num_female_test}\n ")


# ------------------- FEATURE ANALYSIS ------------------- #
def plot_histogram(data, labels):
    # Calcola la media per ciascuna feature
    mean_values = np.mean(data, axis=1, keepdims=True)

    # Normalizza i dati sottraendo le medie solo se la media è diversa da zero
    if np.any(mean_values != 0):
        normalized_data = data - mean_values
    else:
        normalized_data = data

    # Estrai i dati per maschi e femmine in base alle etichette (0 per maschi, 1 per femmine)
    male_data = normalized_data[:, labels == 0]
    female_data = normalized_data[:, labels == 1]

    num_features = data.shape[0]  # Ottieni il numero di feature

    for feature_index in range(num_features):
        # Crea una nuova figura e un asse per ciascuna feature
        plt.figure()

        # Imposta le etichetta
        feature_name = "Feature " + str(feature_index + 1)
        title = "Probability Density - " + feature_name
        plt.title(title, fontsize=10, fontweight="bold")

        
        # Crea gli istogrammi per maschi e femmine
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

        # Aggiungi la legenda
        plt.legend()

        # Regola il layout
        plt.tight_layout()

        # Salva il grafico in un file PDF con un nome basato sull'indice della feature
        plt.savefig("Dataset/Analysis/Histograms/histogram{}.pdf".format(feature_index))

def plot_scatter_matrix(data, labels):
    # Separate Male and Female data
    male_data = data[:, labels == 0]
    female_data = data[:, labels == 1]

    num_features = data.shape[0]  # Ottieni il numero di feature

    # Crea un nuovo grafico vuoto
    plt.figure()

    # Itera su tutte le combinazioni di coppie di feature
    for feature_index1 in range(num_features):
        for feature_index2 in range(feature_index1 - 1, -1, -1):
            feature_name1 = f"Feature {feature_index1 + 1}"
            feature_name2 = f"Feature {feature_index2 + 1}"

            plt.xlabel(feature_name1)
            plt.ylabel(feature_name2)

            plt.scatter(male_data[feature_index1, :], male_data[feature_index2, :],
                        label="Male", color="black", alpha=0.9)
            plt.scatter(female_data[feature_index1, :], female_data[feature_index2, :],
                        label="Female", color="orange", alpha=0.9)
            plt.legend()
            plt.tight_layout()

            # Salva il grafico in un file PDF con un nome basato sugli indici delle feature
            plt.savefig(f"Dataset/Analysis/Scatter/Plot_scatter{feature_index1}_{feature_index2}.pdf")

            # Cancella il grafico corrente per liberare memoria per il prossimo
            plt.clf()

    # Chiudi tutti i grafici una volta che sono completi
    plt.close()



def plot_LDA_hist(dataset, label, m):
    # Calculates the first m discriminant components
    W1 = LDA1(dataset, label, m)
    # projection in the new dimensional space defined by the discriminative components
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


def plot_heatmap(dataset, label, cmap_name, ):

    dataset = dataset[:, label]

    heatmap = numpy.zeros((dataset.shape[0], dataset.shape[0]))

    for i in range(dataset.shape[0]):
        for j in range(i + 1):
            coef = abs(pearsonr(dataset[i, :], dataset[j, :])[0])
            heatmap[i][j] = coef
            heatmap[j][i] = coef

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, cmap=cmap_name)

    ax.set_xticks(numpy.arange(dataset.shape[0]))
    ax.set_yticks(numpy.arange(dataset.shape[0]))
    ax.set_xticklabels(numpy.arange(1, dataset.shape[0] + 1))
    ax.set_yticklabels(numpy.arange(1, dataset.shape[0] + 1))

    ax.set_title("Heatmap of Pearson Correlation")
    fig.colorbar(im)

    plt.savefig(filename)
    plt.close(fig)


def plot_features_histograms(DTR, LTR, _title):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    for i in range(12):
        labels = ["male", "female"]
        title = _title + str(i)
        plt.figure()
        plt.title(title)

        y = DTR[:, LTR == 0][i]
        plt.hist(y, bins=60, density=True, alpha=0.4, linewidth=1.0, color='red', edgecolor='black',
                 label=labels[0])
        y = DTR[:, LTR == 1][i]
        plt.hist(y, bins=60, density=True, alpha=0.4, linewidth=1.0, color='blue', edgecolor='black',
                 label=labels[1])
        plt.legend()
        plt.savefig('./images/hist_' + title + '.svg')
        plt.show()


#  -------------- ANALYSIS EXE -------------- #

#plot_histogram(training_data, training_label)
plot_scatter_matrix(training_data, training_label)
# plot_LDA_hist(training_data, training_label, 1)

