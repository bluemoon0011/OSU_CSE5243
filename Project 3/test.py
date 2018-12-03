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
import collections
word=[['a','b'],['a','d']]
sentence=['ab','ac','b','c']
print('a' in sentence)
additionalWordIndex=[2,3]
a=[1,0,1]
b=[1,0,0]
keys=[12,11,89]
values=[1,0,1]
print(values[1:3])
dictionary = dict(zip(keys, values))
print(dictionary)
od = collections.OrderedDict(sorted(dictionary.items()))
print(od)
'''
for w in word:
    print((w[0] in sentence) and (w[1] in sentence))
'''
#print((word[0] in sentence) and (word[1] in sentence) and (word[2] in sentence))