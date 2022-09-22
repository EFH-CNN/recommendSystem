import pandas as pd
import numpy


def readCSV(path, trainFile=True):
    if trainFile:
        train = pd.read_csv(path)
        user = list(set(train['userID']))
        item = list(set(train['itemID']))

        prefer_matrix = pd.pivot_table(train, index='userID', columns='itemID', values=['rating'], fill_value=0.0)
        prefer_matrix = prefer_matrix.values
        return user, item, prefer_matrix


def read_test_CSV(path):
    test = pd.read_csv(path)
    user = list(set(test['userID']))
    item = list(set(test['itemID']))
    test_matrix = test.values
    return user, item, test_matrix
