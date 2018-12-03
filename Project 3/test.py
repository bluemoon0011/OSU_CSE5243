#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xin Jin
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: test.py
@time: 12/2/18 6:12 PM
@desc:
'''
import numpy as np
word=[['a','b'],['a','d']]
sentence=['ab','ac','b','c']
print('a' in sentence)
additionalWordIndex=[2,3]
del sentence[additionalWordIndex[0]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
a= np.delete(a, [1,2], 0)
print(a)
print('a'+' '+'b')
'''
for w in word:
    print((w[0] in sentence) and (w[1] in sentence))
'''
#print((word[0] in sentence) and (word[1] in sentence) and (word[2] in sentence))