from data_loader import loadTaskData, loadTrainData, loadMovieDetails
import numpy as np
from collections import Counter
import random
import pandas as pd

def main():
    trainData = loadTrainData()
    movieDetails = loadMovieDetails()
    
    k = 38
    f = open("submission.csv","w+")

    taskData = loadTaskData()
    for index, row in taskData.iterrows():
        personId = row['personId']
        movieId = row['movieId']
        investigatedMovie = movieDetails[movieId]
        
        personChoices = trainData.loc[trainData['personId'] == personId]
        similarities = []
        for idx, choice in personChoices.iterrows():
            knownMovie = movieDetails[choice['movieId']]
            similarityScore = similarity(knownMovie, investigatedMovie)
            similarities.append((similarityScore, choice['review']))

        similarities.sort(reverse = True)
        nearestNeighbors = similarities[:k]
        nearestLabels = [x[1] for x in nearestNeighbors]
        review = max(set(nearestLabels), key=nearestLabels.count)

        # if index == 0:
            # break

        stringRow = f"{int(row['id'])};{int(personId)};{int(movieId)};{review}\n"
        f.write(stringRow)

    f.close()


def test():
    trainData = loadTrainData()
    movieDetails = loadMovieDetails()
    
    trainData, validationData = splitData(trainData)
    # taskData = loadTaskData()

    diffs = [ [0] * 6 for _ in range(61)]
    count = 0
    
    print("starting iterations")
    for index, row in validationData.iterrows():
        personId = row['personId']
        movieId = row['movieId']
        investigatedMovie = movieDetails[movieId]
        
        personChoices = trainData.loc[trainData['personId'] == personId]
        similarities = []
        for idx, choice in personChoices.iterrows():
            knownMovie = movieDetails[choice['movieId']]
            similarityScore = similarity(knownMovie, investigatedMovie)
            similarities.append((similarityScore, choice['review']))

        similarities.sort(reverse = True)

        for k in range(1, 61):
            nearestNeighbors = similarities[:k]
            nearestLabels = [x[1] for x in nearestNeighbors]
            review = max(set(nearestLabels), key=nearestLabels.count)
            
            diff = abs(row['review'] - review)
            diffs[k][diff] += 1
        count += 1
        
    
    for k in range(1, 61):
        print(f"{k}   {diffs[k][0]/count}   {(diffs[k][0]+diffs[k][1])/count}")

def splitData(data):
    colnames = ['id', 'personId', 'movieId', 'review']
    validation = pd.DataFrame(columns = colnames)
    train = pd.DataFrame(columns = colnames)

    for idx, row in data.iterrows():
        if idx % 4 == 0:
            validation.loc[len(validation)] = row
        else:
            train.loc[len(train)] = row
        
    return train, validation

def similarity(movie1, movie2):
    sameGenresCount = len(list(set(movie1.genres).intersection(movie2.genres)))
    sameKeywordsCount = len(list(set(movie1.keywords.keywords).intersection(movie2.keywords.keywords)))
    languagesMatchPoint = 1 if movie1.original_language == movie2.original_language else 0
    voteDifference = (movie1.vote_average - movie2.vote_average) * (movie1.vote_average - movie2.vote_average)

    similarity = 3 * sameGenresCount + 2 * sameKeywordsCount + languagesMatchPoint - 2 * voteDifference
    return similarity


if __name__ == '__main__':
    main()