import readFile
import numpy
import csv

user, item, preference_matrix = readFile.readCSV('train.csv')
item_user = preference_matrix.T


def get_user_average_rating():
    return preference_matrix.sum(1) / (preference_matrix != 0).sum(1)


def get_item_similarity_matrix():
    item_similarity = numpy.corrcoef(item_user)
    # col = len(item)
    # item_similarity = numpy.zeros((col, col))
    # for i in range(col):
    #     for j in range(col):
    #         if i <= j < col:
    #             sim_i_j = numpy.dot(item_user[i], item_user[j]) / (
    #                     numpy.linalg.norm(item_user[i]) * (numpy.linalg.norm(item_user[j])))
    #             item_similarity[i][j] = sim_i_j
    #             item_similarity[j][i] = sim_i_j
    return item_similarity


def prediction(item_similarity_matrix, userId, itemId):
    result = get_user_average_rating()[userId]
    numerator = 0.0
    denominator = 0.0
    m = item.index(itemId)
    item_n = len(item)
    for i in range(item_n):
        if item_user[i][userId] != 0.0:
            numerator += item_similarity_matrix[m][i] * item_user[i][userId]
            denominator += numpy.abs(item_similarity_matrix[m][i])
    if denominator != 0.0:
        result = numerator / denominator
        if result <= 0.0:
            result = get_user_average_rating()[userId]
    return result


if __name__ == '__main__':
    user, item, test_index = readFile.read_test_CSV('test_index.csv')
    m = len(test_index)
    item_similarity_matrix = get_item_similarity_matrix()
    number = []
    rating = []
    number.append('dataID')
    rating.append('rating')
    for i in range(m):
        number.append(i)
        rating.append(prediction(item_similarity_matrix, test_index[i][0], test_index[i][1]))
        print(i)

    with open('out_4.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(number, rating))


