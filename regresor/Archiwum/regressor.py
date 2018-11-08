from file_utils import import_trainset, import_testset
from itertools import combinations_with_replacement
import random

#import matplotlib.pyplot as plt

def main():
    #Load data
    trainSet = import_trainset()
    #Preprocessing
    inputs, expected = splitTrainSet(trainSet)
    level = 2 #TODO: Find level

    expandedInputs = [expand_input(trainCase, level) for trainCase in inputs]
    expandedInputs, mins, maxs = normalize_input(expandedInputs)

    model = Model()
    errors = model.train(inputs=expandedInputs, expected=expected, iterations=1000000, alpha = 0.1, desiredError = 0.001, momentum=0.7)
    #plt.figure(1)
    #plt.plot(errors)
    # plt.show()

    #SHOW PREDICTION
    # trainSet = import_trainset()
    # inputs, expected = splitTrainSet(trainSet)
    # expandedInputs = [expand_input(trainCase, level) for trainCase in inputs]
    # x = []
    # y = []
    # for trainCase in expandedInputs:
    #     x.append(trainCase[0])
    #     y.append(model.predict(trainCase, False, mins, maxs))
    
    # plt.figure(2)
    # plt.plot(x, y)
    # plt.plot(inputs, expected)
    # plt.show()

    testSet = import_testset()
    for case in testSet:
        expanded = expand_input(case, 2)
        prediction = model.predict(expanded, False, mins, maxs)
        print(prediction)

def normalize_input(input):
    arr = []
    columns = [column(input, i) for i, f in enumerate(input[0])]
    mins = [min(col) for col in columns]
    maxs = [max(col) for col in columns]
    avgs = [sum(col)/len(col) for col in columns]

    for line in input:
        linearr = []
        for idx, elem in enumerate(line):
            linearr.append((elem - mins[idx])/(maxs[idx] - mins[idx]))
        arr.append(linearr)
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
    
    def predict(self, input, isNormalized, mins, maxs):
        sum = self.p[0]
        for idx, val in enumerate(input):
            value = val if isNormalized else (val - mins[idx])/(maxs[idx]-mins[idx])
            sum += self.p[idx+1] * value
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
                prediction = self.predict(inputCase, True, [], [])
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