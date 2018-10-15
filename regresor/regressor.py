import sys
import numpy as np
from file_utils import import_trainset

def main():
    trainSet = import_trainset()
    #TODO: learn

    #test
    testSet = import_testset()
    for case in testSet:
        prediction = predict(case)
        print(prediction)
    

def import_testset():
    floats = []
    for line in sys.stdin:
        floats.append([float(x) for x in line.split()])
    return np.array(floats)

def predict(case):
    return 0

if __name__ == "__main__":
    main()