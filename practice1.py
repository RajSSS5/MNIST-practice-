from __future__ import division
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.multiclass import OneVsRestClassifier
import sklearn.linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
# Read in values from csv file and load into DataFrame type from pandas
print("Reading training data")
trainSet = pd.read_csv(r"Data\train.csv")
print("Done Reading")
# Drop vals
# trainSet.dropna(inplace=True)

# Replace vals with mean
# trainSet.fillna(trainSet.median(), inplace=True)

mat = np.array(trainSet)

y = trainSet.iloc[:,0]
X = trainSet.iloc[:,1:]

# SVMS ARE NOT SCALE INVARIANT!

print("PCA comp")
pca = decomposition.PCA(n_components=50)
pca.fit(X)
X = pca.transform(X)
print("Done with PCA comp")
#model = RandomForestClassifier() 
#model = sklearn.linear_model.LogisticRegression(multi_class='ovr')
# model = DecisionTreeClassifier() 70%
print("Fitting")
model = GaussianNB()
#model =  svm.SVC(gamma='scale')
#model =  svm.LinearSVC()
model.fit(X,y)
print("Done Fitting")
# File with all test data
print("Reading test")
testSet = pd.read_csv(r"Data\test.csv")
print("Done Reading")
X_test = testSet #.iloc[1:]
print("PCA comp")
pca = decomposition.PCA(n_components=50)
pca.fit(X_test)
X_test = pca.transform(X_test)
print("Done PCA comp")

dataList = []
print("Predicting")
for item in X_test:
        item = np.array(item).reshape(1,-1)
        res = model.predict(item)
        dataList.append(*res) 
print("Done Predicting")


indexList = [i for i in range(1,len(dataList) + 1)]

dataDict = {'ImageId': indexList, 'Label': dataList}

dataDF = pd.DataFrame.from_dict(dataDict)

dataDF.to_csv(path_or_buf=r"Data\predictions.csv",mode='w',index=False)

# Naive Bayes seems to get the best result vs. time
# ================================================
# PCA with 154 components got a result of 0.5417
# PCA with 50 cpmponents got a resuly of .50542

# Linear SVC => ~40% with PCA of 154, took much longer 