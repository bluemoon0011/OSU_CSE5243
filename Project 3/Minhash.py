#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xin Jin
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: Minhash.py
@time: 12/2/18 11:19 PM
@desc:
'''
import random
import pandas as pd
import collections
import numpy

def pickRandomCoeffs(k):
    # Create a list of 'k' random values.
    randList = []
    maxShingleID=2**32-1

    while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID)

        # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, maxShingleID)

            # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1

    return randList

def getPermutation(numHashes):
    nextPrime = 4294967311
    numShingle = 7344
    coeffA = pickRandomCoeffs(numHashes)
    coeffB = pickRandomCoeffs(numHashes)
    Permutation = []
    for i in range(numHashes):
        signature = []
        for j in range(numShingle):
            hashcode = (coeffA[i] * j + coeffB[i]) % nextPrime
            signature.append(hashcode)
        '''for i in range(numShingle):
            for j in range(i + 1, numShingle):
                if signature[i] == signature[j]:
                    print('false')'''
        Permutation.append(signature)
    return Permutation

def getMinHash(Permutation):
    dataset = pd.read_csv('rawresult.csv', sep=',')
    matrix = dataset.values
    minHash = numpy.zeros((len(matrix), numHashes), dtype=int)
    for i in range(len(matrix)):
        sentence=matrix[i]
        print('Round of %d' % i)
        for j in range(numHashes):
            hash=Permutation[j]
            keys=hash
            values=sentence
            dictionary=dict(zip(keys, values))
            od = collections.OrderedDict(sorted(dictionary.items()))
            for key in od.keys():
                if od[key]==1:
                    minHash[i,j]=key
                    break
    return minHash
# For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.


if __name__ == "__main__":
    numHashes=128
    numShingle = 7344
    Permutation=getPermutation(numHashes)
    minHash=getMinHash(Permutation)
    df=pd.DataFrame(minHash)
    df.to_csv('minHash.csv')
