from data_loader import loadTaskData, loadTrainData, loadMovieDetails
import numpy as np
from collections import Counter
import random
import pandas as pd
import ast
import math

def main():
    trainData = loadTrainData()
    taskData = loadTaskData()
    uniquePersons = trainData['personId'].unique()
    similaritiesFile= open('similarities.txt', 'r')
    similarities = similaritiesFile.read()
    similaritiesFile.close()
    similarities = ast.literal_eval(similarities)
    # print(similarities)
    # return
    # print(len(uniquePersons))
    # index = 0
    # for superPersonId in uniquePersons:
    #     superPersonRatings = trainData.loc[trainData['personId'] == superPersonId]
    #     similarities[superPersonId] = []
    #     for personId in uniquePersons:
    #         if personId == superPersonId:
    #             similarities[superPersonId].append(0)
    #             continue
    #         personRatings = trainData.loc[trainData['personId'] == personId]
    #         # similarity = -1
    #         similarity = calcSimilarity(superPersonRatings, personRatings)
    #         # break
    #         similarities[superPersonId].append(similarity)
    #     # break
    #     print(index)
    #     index += 1

    # f = open("similarities.txt","w")        
    # f.write(str(similarities))
    # f.close()
    # return
    # for each test case
    result = ''
    counter = 0
    for index, row in taskData.iterrows():
        counter +=1
        if counter <= 1900:
            continue
        personId = row['personId']
        movieId = row['movieId']
        
        similartiesWeightSum = 0
        predictionSum = 0
        filteredByMovie = trainData.loc[(trainData['movieId'] == movieId)]
        for idx, sim in enumerate(similarities[personId]):
            checkedPersonId = uniquePersons[idx]
            if checkedPersonId == personId:
                continue
            
            reviewRow = filteredByMovie.loc[(filteredByMovie['personId']==checkedPersonId)]
            if reviewRow.empty:
                continue
            review = reviewRow['review'].values[0]
            predictionSum += review * sim
            similartiesWeightSum += sim
        
        finalReview = predictionSum / similartiesWeightSum
        if math.isnan(finalReview):
            finalReview = 3
        else:
            finalReview = int(finalReview.round())
        
        stringRow = f"{int(row['id'])};{int(personId)};{int(movieId)};{finalReview}\n"
        result += stringRow

        if counter%100==0:
            print(counter)
            f = open("submission.csv","a+")
            f.write(result)
            f.close()
            result = ''
            # break
    print(counter)
    f = open("submission.csv","a+")
    f.write(result)
    f.close()

def calcSimilarity(superPersonRatings, personRatings):
    # for each rating of a super person
    superFiltered = superPersonRatings[superPersonRatings['movieId'].isin(personRatings['movieId'])]
    personFiltered = personRatings[personRatings['movieId'].isin(superFiltered['movieId'])]
    superFiltered = superFiltered.sort_values('movieId')
    personFiltered = personFiltered.sort_values('movieId')

    diffs = superFiltered['review'].values - personFiltered['review'].values
    avgDiff = sum(abs(i) for i in diffs)/len(diffs)
    if avgDiff > 1.5:
        return 0

    sumSquared = sum(i*i for i in diffs)
    
    return len(diffs)/(sumSquared+1)

if __name__ == '__main__':
    main()