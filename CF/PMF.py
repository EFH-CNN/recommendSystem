import numpy as np
import readFile
import csv

user, item, preference_matrix = readFile.readCSV('train.csv')


def create_random_matrix(useritem_n, k):
    return np.random.random(size=(useritem_n, k))


def get_user_average_rating():
    return preference_matrix.sum(1) / (preference_matrix != 0).sum(1)


def ALS(P, Q, ratings, type='user', k=20, E=0.001):
    lambdaE = np.eye(k) * E
    if type == 'user':
        for i in range(len(user)):
            Ai = np.dot(Q.T, Q) + lambdaE
            # print(Ai)
            Vi = np.dot(ratings[i, :].T,Q)
            # print(Vi)
            P[i, :] = np.linalg.solve(Ai, Vi)
            # print(latent_matrix)

    if type == 'item':
        for i in range(len(item)):
            Ai = np.dot(Q.T, Q) + lambdaE
            Vi = np.dot(ratings[:, i].T,Q)
            P[i, :] = np.linalg.solve(Ai + lambdaE, Vi)

    return P


def train(iter):
    user_k = create_random_matrix(len(user), 20)
    item_k = create_random_matrix(len(item), 20)
    times = 0
    while times < iter:
        user_k = ALS(user_k, item_k, preference_matrix, type='user')
        item_k = ALS(item_k, user_k, preference_matrix, type='item')
        times += 1
    return user_k.dot(item_k.T)


if __name__ == '__main__':
    a = train(200)
    b = get_user_average_rating()
    # print(b)
    # print(a[2346][item.index(468)])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] < 1.0:
                a[i][j] = b[i]
            if a[i][j] > 5.0:
                a[i][j] = 5.0

    user, item, test_index = readFile.read_test_CSV('test_index.csv')
    m = len(test_index)
    number = []
    rating = []

    number.append('dataID')
    rating.append('rating')
    for i in range(m):
        number.append(i)
        rating.append(a[test_index[i][0]][item.index(test_index[i][1])])
        print(i)

    print(rating)

    with open('out_3.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(number, rating))
