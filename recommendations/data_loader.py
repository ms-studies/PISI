import pandas as pd
import urllib.request
import json
from collections import namedtuple
from time import sleep


def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

def loadMoviesData():
    colnames = ['id', 'webid', 'name']
    data = pd.read_csv('data/movie.csv', names = colnames, sep=';')
    return data

def loadTaskData():
    colnames = ['id', 'personId', 'movieId', 'review']
    data = pd.read_csv('data/task.csv', names = colnames, sep=';')
    return data

def loadTrainData():
    colnames = ['id', 'personId', 'movieId', 'review']
    data = pd.read_csv('data/train.csv', names = colnames, sep=';')
    return data

def loadMovieDetails():
    cols = ['intId', 'webId', 'details']
    data = pd.read_csv('data/movie_details.csv', names = cols, sep=';')
    map = {}
    for index, row in data.iterrows():
        map[row['intId']] = json2obj(row['details'])
    
    return map

def getDataFromApi(data):
    cols = ['intId', 'webId', 'details']
    apidataframe = pd.DataFrame(columns = cols)
    for index, row in data.iterrows():
        id = row['webid']
        contents = urllib.request.urlopen(f"https://api.themoviedb.org/3/movie/{id}?api_key=42aa0017f58528179d15a0ec047d7a26&language=en-US&append_to_response=keywords").read().decode('utf-8')
        apidataframe.loc[len(apidataframe)] = [row['id'], row['webid'], contents]
        sleep(0.3)

    apidataframe.to_csv('data/movie_details.csv', sep=';', index=False, header=False)

    return data

def main():
    data = loadMoviesData()
    data = getDataFromApi(data)

if __name__ == '__main__':
    main()