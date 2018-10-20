#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xin Jin
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: classification.py
@time: 10/9/18 7:00 PM
@desc:
'''

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import types
import re
import operator
from itertools import islice
from nltk.corpus import stopwords
from operator import itemgetter
import KNN as knn

def stringTofloat(list):
    for i in range(len(list)):
        l=list[i]
        l = re.sub("\D", "", l)
        l=float(l)
        list[i]=l
    return list

def splitDataset(attributeset,labelset,splitsize):
    attributeset_train, attributeset_test, labelset_train, labelset_test = train_test_split(attributeset,labelset,test_size=splitsize)
    return attributeset_train,attributeset_test,labelset_train,labelset_test

def computeTF_IDF(trainingSet,feature):
    tfScore=np.zeros(len(feature),dtype=float)
    for i in range(len(feature)):
        value=trainingSet[feature[i]]
        tf=sum(value)
        numContainFeature=np.count_nonzero(np.shape(value))
        idf=np.log2(3000.0/numContainFeature)
        tfScore[i]=tf*idf
    return list(tfScore)

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def chooseKeyFeature(tfScore,feature,N): # this is used to choose N features with the larger tfScore
    dictionary = dict(zip(feature, tfScore))
    sorted_list= sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    keyFeature=list()
    for i in range(N):
        keyFeature.append(sorted_list[i][0])
    return keyFeature

def getKeyFeature(originalDataset,trainingDataset,N):
    feature=originalDataset.columns.values.tolist()
    feature = feature[1:]
    filteredFeature = [word for word in feature if word not in stopwords.words('english')]
    keyFeature = chooseKeyFeature(computeTF_IDF(trainingDataset, filteredFeature), filteredFeature, N)
    #filtered_keyFeature = [word for word in keyFeature if word not in stopwords.words('english')]
    return keyFeature

def Tup():
  return (3,"hello")

"*** Main function***"

dataset = pd.read_csv('original_dataset.csv', sep='\t', names=['sentence', 'label', ]) # read the origianl text dataset which contains 3000 sentences and labels
label = stringTofloat(dataset['label'])# the list of labels
resultset = pd.read_csv('result_of_project1.csv', sep=',')# read the dataset, which is the resulf of Project 1
feature=resultset.columns.values.tolist() # get the column of the dataFrame, which is the feature of the dataset
feature=feature[1:] # exclude the first element of the feature vector, which is the name of the index
trainingSet,testSet0,trainingLabel,testLabel0=splitDataset(resultset,label,0.4)# get 60% as test set/label, 40% as raw test set/label
testSet,validationSet,testLabel,validationLabel=splitDataset(testSet0,testLabel0,0.5)# split the raw test set as test and validation set, and the test and validation set size is 20% and 20%
keyFeature=getKeyFeature(resultset,trainingSet,1000)
trainingSet_pruned=trainingSet[keyFeature]
testSet_pruned=testSet[keyFeature]
trainingSet_unpruned=trainingSet[feature]
testSet_unpruned=testSet[feature]


"*** Classify by KNN ***"
#test the accuracy rate of the pruned(prune the stop word) feature
labels=knn.ClassifyTestset(trainingSet_pruned.as_matrix(),testSet_pruned.as_matrix(),trainingLabel.as_matrix(),testLabel.as_matrix(),8)
rate=knn.ComputeAccuracy(labels,testLabel.as_matrix())
print rate

#test the accuracy rate of the unpruned feature
labels1=knn.ClassifyTestset(trainingSet_unpruned.as_matrix(),testSet_unpruned.as_matrix(),trainingLabel.as_matrix(),testLabel.as_matrix(),8)
rate=knn.ComputeAccuracy(labels,testLabel.as_matrix())
print rate
"*** Test Code ***"
'''
np_testset = testSet_pruned.as_matrix()
np_trainingset = trainingSet_pruned.as_matrix()
print np_testset[0]

filtered_words = [word for word in word_list if word not in stopwords.words('english')]
tfScore=[11,2,6,4]
feature=['so','you','are','ok']
a=chooseKeyFeature(tfScore,feature,2)
print a
tfScore=np.array(tfScore)
feature=np.array(feature)
a=np.vstack((feature,tfScore))
a=list(a)
sorted(a, key=itemgetter(1))
N=2
matrix = [ [4,5,6], [1,2,3], [7,0,9]]
sorted(matrix, key=itemgetter(1))
#print np.array(feature)
#print chooseKeyFeature(tfScore,feature,N)
'''
"""
filtered_words = [word for word in feature if word not in stopwords.words('english')]
filtered_feature= [word for word in feature if word not in stopwords.words('english')] # 

print len(label)

X, y = np.arange(10).reshape((5, 2)), range(5)

print("X:{} \n", X)
#print("y: {} \n", list(y))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train,X_test=train_test_split(X,test_size=0.33, random_state=100)

print("X_train:{}\n", X_train)
print("X_test:{}\n", X_test)
#print("y_train:{} \n", y_train)
#print("y_test:{}\n", y_test)

#print(train_test_split(y, shuffle=False))
"""