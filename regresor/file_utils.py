import argparse
import numpy as np

def import_trainset():
    path = read_train_path()
    return import_trainset_from_path(path)

def import_trainset_from_path(path):
    with open(path) as f:
        content = f.readlines()
    lines = [x.strip() for x in content]
    floats = []
    for line in lines:
        floats.append([float(x) for x in line.split()])
    floats = np.array(floats)
    lastIndex = len(floats[0])-1
    x = floats[:, 0:lastIndex]
    y = floats[:, lastIndex]
    return x, y

def read_train_path():
    parser=argparse.ArgumentParser()
    parser.add_argument('-t', help='Dataset path')
    args=parser.parse_args()
    return args.t