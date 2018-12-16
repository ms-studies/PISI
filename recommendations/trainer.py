from data_loader import loadTaskData, loadTrainData, loadMovieDetails
import numpy as np
from collections import Counter

def main():
    trainData = loadTrainData()
    movieDetails = loadMovieDetails()
    
    
    k = 8
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

        stringRow = f"{int(row['id'])};{int(personId)};{int(movieId)};{review}\n"
        f.write(stringRow)

    f.close()
 

def similarity(movie1, movie2):
    sameGenresCount = len(list(set(movie1.genres).intersection(movie2.genres)))
    languagesMatchPoint = 1 if movie1.original_language == movie2.original_language else 0
    # popularityMatch = 
    similarity = 3 * sameGenresCount + languagesMatchPoint
    return similarity


if __name__ == '__main__':
    main()