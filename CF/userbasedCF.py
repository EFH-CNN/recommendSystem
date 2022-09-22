import numpy as np
import readFile
import pandas as pd
import csv

user, item, preference_matrix = readFile.readCSV('train.csv')


def get_user_average_rating():
    return np.sum(preference_matrix, 1) / np.sum((preference_matrix != 0), 1)


def similarity():
    similarity_matrix = np.corrcoef(preference_matrix)
    return similarity_matrix


similarity_matrix1 = similarity()


def similarity_k_most(userId, k):
    k_similarity = [(similarity_matrix1[userId][other], other) for other in
                    range(len(user))
                    if other != userId]
    k_similarity.sort(reverse=True)
    similarity_person = []
    similarity_userId_k = []
    for sim in range(k):
        similarity_userId_k.append(k_similarity[sim][1])
        similarity_person.append(k_similarity[sim][0])
    return similarity_userId_k, similarity_person


def prediction(userId, itemId):
    neighbors, sim = similarity_k_most(userId, 20)
    average = get_user_average_rating()
    numerator = 0.0
    denominator = 0.0
    result = get_user_average_rating()[userId]
    # print(preference_matrix[neighbors[0]])
    # print(average[neighbors[0]])

    for i in range(len(neighbors)):
        if preference_matrix[neighbors[i]][item.index(itemId)] != 0:
            numerator += sim[i] * (preference_matrix[neighbors[i]][item.index(itemId)] - average[neighbors[i]])
            denominator += abs(sim[i])
        else:
            continue

    if denominator != 0:
        result = average[userId] + (numerator / denominator)
    return result


# def get_rmse(userId, prediction_result):
#     sum1 = 0.0
#     number = 0
#     for i in range(len(item)):
#         if preference_matrix[userId][i] != 0:
#             sum1 += pow(preference_matrix[userId][i] - prediction_result[i], 2)
#             number += 1
#
#     return pow(sum1 / number, 0.5)


if __name__ == '__main__':
    user,item,test_index = readFile.read_test_CSV('test_index.csv')
    m = len(test_index)
    number = []
    rating = []
    number.append('dataID')
    rating.append('rating')
    for i in range(m):
        number.append(i)
        rating.append(prediction(test_index[i][0], test_index[i][1]))
        print(i)

    with open('out_1.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(number, rating))



