import numpy
from utils import *


def load(fname):
    DList = []
    labelsList = []
    hLabels = {"0": 0, "1": 1}

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(",")[0:12]  
                attrs = mcol(numpy.array([float(i) for i in attrs]))  
                name = line.split(",")[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except: 
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)