from Functions.reshape_functions import *

def znorm(dataset):
    mean = mcol(dataset.mean(1))
    std = mcol(dataset.std(1))
    dataset = (dataset - mean) / std
    return dataset
