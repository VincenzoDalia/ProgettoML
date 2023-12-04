import numpy
from Functions.reshape_functions import *


def load(fname):
    # dataList contains the attributes during the elaboration of the file
    dataList = []
    # labelsList contains the labels during the elaboration
    labelsList = []
    # LabelMapping parse from string to int
    labelMapping = {"0": 0, "1": 1}

    try:
        with open(fname, 'r') as file:
            for line in file:
                # Split the line using a comma as the separator
                elements = line.strip().split(',')

                # The first 12 elements are attributes and convert them to float
                attributes = numpy.array([float(element)
                                         for element in elements][0:12])

                attributes = mcol(attributes)

                # Extract the label, which is the last element on the line
                label = elements[-1].strip()

                # Map the label to its corresponding numeric value using labelMapping
                # else generate an exception
                if label in labelMapping:
                    label = labelMapping[label]
                else:
                    raise ValueError("Invalid label: " + label)

                # Append the attributes and label to their respective lists
                dataList.append(attributes)
                labelsList.append(label)

    except IOError as e:
        raise IOError("An error occurred while reading the file: " + str(e))
    except ValueError as e:
        raise ValueError("An error occurred while processing data: " + str(e))

    # Stack the attribute arrays horizontally to form a 2D array
    attributes = numpy.hstack(dataList)

    # Convert the labels list to a numpy array with dtype int32
    labels = numpy.array(labelsList, dtype=numpy.int32)

    return attributes, labels
