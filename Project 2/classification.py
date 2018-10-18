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


dataset = pd.read_csv('dataset.csv', sep='\t', names=['sentence', 'label', ])

for i in range(len(dataset['label'])):
    label=dataset['label'][i]
    if label == 1:
        pass
    elif label == 0:
        pass
    elif label == '0"':
        pass
    elif label == '1"':
        pass
    else:
        print label, i



print dataset['label']

"""
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