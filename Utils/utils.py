import numpy
import matplotlib as plt
import scipy

def mcol(array):
    return array.reshape((array.size, 1))


def mrow(array):
    return array.reshape((1, array.size))


# --------------- LDA utils functions --------------- #

# Function to calculate the inter-class covariance
def Sw_c(dataset):
    Sw_c = 0
    nc = dataset.shape[1]
    mu_c = mcol(dataset.mean(1))
    centeredDataset = dataset - mu_c
    # Dot product between the dataset and its transpose over the number of samples
    Sw_c = numpy.dot(centeredDataset, centeredDataset.T) / nc
    return Sw_c


#Function to calcolate the within-class covariance matrix Sw and the between-class covariance matrix Sb
def SbSw(matrix, label):
    Sb = 0
    Sw = 0
    mu = mcol(matrix.mean(1))
    N = matrix.shape[1]
    for i in range(label.max() + 1):
        D_c = matrix[:, label == i]
        nc = D_c.shape[1]
        mu_c = mcol(D_c.mean(1))
        Sb = Sb + nc * numpy.dot((mu_c - mu), (mu_c - mu).T)
        Sw = Sw + nc * Sw_c(D_c)
    Sb = Sb / N
    Sw = Sw / N

    return Sb, Sw

# LDA Function to calculate the discriminant elements, obtained analyzing Sw and Sb
#the parameter m indicates the number of discriminant elements we have to generate
def LDA1(matrix, label, m):
    Sb, Sw = SbSw(matrix, label)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    return W


