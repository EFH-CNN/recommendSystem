import csv

import pandas as pd

with open("API.txt", "r") as f:  # 打开文件
    data = f.read()  # 读取文件

api = data.split(',')
# print(API)

df = pd.read_csv('APIGu1.csv')
print(df.shape)
# df.drop('android.animation.FloatArrayEvaluator', axis=1, inplace=True)
# df.to_csv("handle.csv", index=0)
shape = df.shape[1]

for i in api:
    print(i)
    df.drop(i, axis=1, inplace=True)
    df.to_csv("handle.csv", index=0)
