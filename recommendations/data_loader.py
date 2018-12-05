import pandas as pd

def main():
    colnames = ['id', 'webid', 'name']

    data = pd.read_csv('movie.csv', names = colnames, sep=';')
    # print(data)

def loadTaskData():
    colnames = ['id', 'personId', 'movieId', 'review']

    data = pd.read_csv('data/task.csv', names = colnames, sep=';')
    return data


if __name__ == '__main__':
    main()