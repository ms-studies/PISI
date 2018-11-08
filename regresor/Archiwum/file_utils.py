import argparse
import sys

def import_trainset():
    path = read_train_path()
    return import_trainset_from_path(path)
    

def import_trainset_from_path(path):
    with open(path) as f:
        content = f.readlines()
    
    arr = []
    for line in content:
        numbers = line.split(' ')
        numline = []
        for strnum in numbers:
            num = float(strnum.strip())
            numline.append(num)
        
        arr.append(numline)

    return arr

def import_testset():
    arr = []
    for line in sys.stdin:
        numbers = line.split(' ')
        numline = []
        for strnum in numbers:
            num = float(strnum.strip())
            numline.append(num)
        
        arr.append(numline)

    return arr

def read_train_path():
    parser=argparse.ArgumentParser()
    parser.add_argument('-t', help='Dataset path')
    args=parser.parse_args()
    return args.t