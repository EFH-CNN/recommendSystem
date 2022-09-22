import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import numpy as np

data = pd.read_csv('API_Permission_select.csv')
label = pd.read_csv('label.csv')

# x = data.columns[1:]
# y = data['label']

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)

print('The accuracy of random forest classifier is', rfc.score(x_test, y_test))
print(classification_report(rfc_y_pred, y_test, digits=5))

fpr, tpr, threshold = metrics.roc_curve(y_test, rfc_y_pred)
roc_auc = metrics.auc(fpr, tpr)
print("Fpr is:  " + str(fpr))
print("AUC is:  " + str(roc_auc))
result = classification_report(rfc_y_pred, y_test, digits=5)
a = result.split()
recall = float(a[-3])
precision = float(a[-4])
F_measure = ((recall * precision) / (recall + precision)) * 2

print("F_measure is: " + str(F_measure))
