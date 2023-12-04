import numpy
import matplotlib as plt
import scipy.linalg  # aggiunto per LDA su windows


def mcol(array):
    return array.reshape((array.size, 1))


def mrow(array):
    return array.reshape((1, array.size))













#  --------------- LDA utils functions --------------- #

def compute_Sb(dataset, label):
    num_features, num_samples = dataset.shape
    sb = numpy.zeros((num_features, num_features))
    class_means = []

    for i in range(2):
        class_samples = dataset[:, label == i]
        class_mean = numpy.mean(class_samples, axis=1, keepdims=True)
        class_means.append(class_mean)

    overall_mean = numpy.mean(dataset, axis=1, keepdims=True)
    dataset = dataset - overall_mean

    for i in range(2):
        class_size = numpy.sum(label == i)
        diff = class_means[i] - overall_mean
        sb += class_size * numpy.dot(diff, diff.T)

    Sb = (1 / num_samples) * sb
    return Sb

def compute_Sw(dataset, label):
    num_features, num_samples = dataset.shape
    mean_classes = []

    for i in range(2):
        class_samples = dataset[:, label == i]
        class_mean = numpy.mean(class_samples, axis=1, keepdims=True)
        mean_classes.append(class_mean)

    Sw = numpy.zeros((num_features, num_features))

    for i in range(2):
        class_size = numpy.sum(label == i)
        class_data = dataset[:, label == i]
        diff = class_data - mean_classes[i]
        Sw += class_size * (1 / class_data.shape[1]) * numpy.dot(diff, diff.T)

    Sw = (1 / num_samples) * Sw
    return Sw


# LDA Function to calculate the discriminant elements, obtained analyzing Sw and Sb
# the parameter m indicates the number of discriminant elements we have to generate
def LDA1(matrix, label, m):
    Sb = compute_Sb(matrix, label)
    Sw = compute_Sw(matrix, label)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    return W


# ------------------ PCA ------------------ #

#Da testare ancora
# Usato per i GMM, Gaussian e Logistic Regression, non per feature analysis
def PCA(data_matrix, m, DTE=None):
    N = data_matrix.shape[1]
    mu = numpy.mean(data_matrix, axis=1, keepdims=True)
    DC = data_matrix - mu
    C = numpy.dot(DC, DC.T) / N
    s, U = numpy.linalg.eigh(C)
    # La prossima istruzione serve a selezionare i primi m autovettori più grandi calcolati in precedenza
    # e li assegna a P
    P = U[:, ::-1][:, 0:m]
    # Proiezione dei dati originali sulla nuova base identificata dagli autovettori selezionati in P
    DP = numpy.dot(P.T, data_matrix)
    # Se DTE (Dati di test) è una matrice fornita allora viene anch'essa
    # proiettata sulla nuova base 
    if DTE is not None:
        DTE = numpy.dot(P.T, DTE)
    return DP, DTE


# ------------------ k-Fold ------------------ #

def kfold(function, k, D, L, compute_scores, eff_prior=None, seed=4):
    SPost_partial = []
    folds = []

    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])

    Label = L[idx]

    fold_size = D.shape[1] // k

    # Divide indices into k-folds
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        folds.append(idx[start:end])

    # If the number of samples is not divisible by K, add the remaining samples to the last fold
    if D.shape[1] % k != 0:
        folds[-1] = numpy.concatenate((folds[-1], idx[k * fold_size :]))

    # Perform Cross validation
    for i in range(k):
        # Choose the i-th fold as the validation fold
        validation_indices = folds[i]
        train_indices = numpy.concatenate([folds[j] for j in range(k) if j != i])
        logSPost = function(
            D[:, train_indices],
            L[train_indices],
            D[:, validation_indices],
            L[validation_indices],
            eff_prior,
        )
        scores = compute_scores(eff_prior, logSPost)
        #scores = model.scores FORSE DA TOGLIERE

        SPost_partial.append(scores)
        # print("end fold:",i)

    S = numpy.hstack(SPost_partial)

    return S, Label
