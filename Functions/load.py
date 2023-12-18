import numpy
from Functions.reshape_functions import *


def load(fname):
    dataList = []
    labelsList = []
    labelMapping = {"0": 0, "1": 1}

    try:
        with open(fname, 'r') as file:
            for line in file:
                elements = line.strip().split(',')

                attributes = numpy.array([float(element) for element in elements][0:12])

                attributes = mcol(attributes)
                label = elements[-1].strip()

                if label in labelMapping:
                    label = labelMapping[label]
                else:
                    raise ValueError("Invalid label: " + label)

                dataList.append(attributes)
                labelsList.append(label)

    except IOError as e:
        raise IOError("An error occurred while reading the file: " + str(e))
    except ValueError as e:
        raise ValueError("An error occurred while processing data: " + str(e))

    attributes = numpy.hstack(dataList)

    labels = numpy.array(labelsList, dtype=numpy.int32)

    return attributes, labels
