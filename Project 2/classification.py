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


def stringTofloat(list):
    for i in range(len(list)):
        l=list[i]
        l = re.sub("\D", "", l)
        l=float(l)
        list[i]=l
    return list

dataset = pd.read_csv('original_dataset.csv', sep='\t', names=['sentence', 'label', ])
label = stringTofloat(dataset['label'])
resultset = pd.read_csv('result_of_project1.csv', sep=',')# read the dataset, which is the resulf of Project 1
feature=resultset.columns.values.tolist() # get the column of the dataFrame, which is the feature of the dataset
feature=feature[1:]# the first element is 'unnamed: 0', which is not the element of the original fearure vector
print feature


"*** Test Code ***"

"""
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