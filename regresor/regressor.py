from file_utils import import_trainset, import_testset
from itertools import combinations_with_replacement
import random

#import matplotlib.pyplot as plt

def main():
    #Load train data
    data = import_trainset()
    level = findModelLevel(data)

    inputs, expected = splitTrainSet(data)

    inputs, mins, maxs = normalize_input(inputs)
    inputs = [expand_input(trainCase, level) for trainCase in inputs]

    model = Model()
    errors = model.train(inputs=inputs, expected=expected, iterations=10000000, alpha = 0.1, desiredError = 0.0001, momentum=0.9)

    #====TEST====
    testSet = import_testset()
    testSet = normalize_with_minmax(testSet, mins, maxs)
    testSet = [expand_input(testCase, level) for testCase in testSet]
    for case in testSet:
        prediction = model.predict(case)
        print(prediction)

def findModelLevel(data):
    MSEMatrix = []
    for i in range(1):
        trainSet, validationSet = splitData(data, 0.25)

        trainInput, trainExpected = splitTrainSet(trainSet)
        validationInput, validationExpected = splitTrainSet(validationSet)

        trainInput, mins, maxs = normalize_input(trainInput)
        validationInput = normalize_with_minmax(validationInput, mins, maxs)

        MSEs = []
        for level in range(10):
            trainInputExpanded = [expand_input(trainCase, level+1) for trainCase in trainInput]
            model = Model()
            model.train(inputs=trainInputExpanded, expected=trainExpected, iterations=5000, alpha = 0.1, desiredError = 0.01, momentum=0.9)
        
            validationInputExpanded = [expand_input(valCase, level+1) for valCase in validationInput]
            error = model.test(validationInputExpanded, validationExpected)
            MSEs.append(error)
            if len(MSEs) > 2 and MSEs[level] > MSEs[level-1] and MSEs[level-1] > MSEs[level-2]:
                break

        MSEMatrix.append(MSEs)
    
    avgMSEs = []
    for idx, v in enumerate(MSEMatrix[0]):
        col = column(MSEMatrix, idx)
        avgMSEs.append(sum(col)/len(col))
    idx = avgMSEs.index(min(avgMSEs))
    return idx+1

def splitData(data, percentage):
    random.shuffle(data)
    itemsToTake = int(percentage * len(data))
    validation = []
    train = []
    for idx, row in enumerate(data):
        if idx < itemsToTake:
            validation.append(row)
        else:
            train.append(row)
    return train, validation

def normalize_with_minmax(input, mins, maxs):
    arr = []
    for line in input:
        linearr = []
        for idx, elem in enumerate(line):
            linearr.append((elem - mins[idx])/(maxs[idx] - mins[idx]))
        arr.append(linearr)
    return arr

def normalize_input(input):
    columns = [column(input, i) for i, f in enumerate(input[0])]
    mins = [min(col) for col in columns]
    maxs = [max(col) for col in columns]
    arr = normalize_with_minmax(input, mins, maxs)
    return arr, mins, maxs

def splitTrainSet(trainSet):
    inputs = []
    expected = []
    for case in trainSet:
        expected.append(case[-1])
        inputs.append([x for x in case[:-1]])
    return inputs, expected

def expand_input(input, level):
    expanded = []
    for i in range(level-1):
        for val in input:
            expanded.append(val**(i+1))
    combinations = list(combinations_with_replacement(input, level))
    for combination in combinations:
        combinationValue = 1
        for combinationElement in combination:
            combinationValue *= combinationElement
        expanded.append(combinationValue)
    return expanded

class Model:
    p = []

    def __init__(self):
        self.p = []

    def test(self, inputs, expected):
        error = 0
        for idx, inputCase in enumerate(inputs):
            prediction = self.predict(inputCase)
            exptectation = expected[idx]
            error += pow(prediction - exptectation, 2)
        mse = error / len(inputs)
        return mse
    
    def predict(self, input):
        sum = self.p[0]
        for idx, val in enumerate(input):
            #value = val if isNormalized else (val - mins[idx])/(maxs[idx]-mins[idx])
            sum += self.p[idx+1] * val
        return sum

    def train(self, inputs, expected, iterations, alpha, desiredError, momentum):
        self.p.append(random.random()*2-1)
        for i in range(len(inputs[0])):
            self.p.append(random.random())

        mean_errors = []
        p_changes = [0]*len(self.p)
        iter = 0
        while (len(mean_errors) == 0 or mean_errors[-1] > desiredError) and iter < iterations:
            #Calculate differences
            differences = []
            for idx, inputCase in enumerate(inputs):
                prediction = self.predict(inputCase)
                exptectation = expected[idx]
                difference = prediction - exptectation
                differences.append(difference)
            mean_errors.append(sum([x*x for x in differences])/len(inputs))
            
            #Calculate errors for p
            errors = []
            for i in range(len(self.p)):
                error = 0
                if i == 0:
                    error = sum(differences) / len(inputs)
                else:
                    error = 0
                    for idx, inputCase in enumerate(inputs):
                        difference = differences[idx]
                        x = inputCase[i-1]
                        error += difference * x
                    error = error / len(inputs)
                errors.append(error)
        

            for idx, error in enumerate(errors):
                change = -alpha*error
                self.p[idx] = self.p[idx] + change + momentum * p_changes[idx]
                p_changes[idx] = change
            
            iter += 1
            #print(self.p)
        return mean_errors

def column(matrix, i):
    return [row[i] for row in matrix]

if __name__ == "__main__":
    main()