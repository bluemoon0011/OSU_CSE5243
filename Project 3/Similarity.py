#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xin Jin
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: Similarity.py
@time: 12/2/18 11:19 PM
@desc:
'''
import pandas as pd
import numpy
from sklearn.metrics import jaccard_similarity_score
import time

def lenInteraction(a,b):
    length=0
    for i in range(len(a)):
        if a[i]==1 and b[i]==1:
            length+=1
    return length
def lenUnion(a,b):
    length=0
    for i in range(len(a)):
        if a[i]==1 or b[i]==1:
            length+=1
    return length
def lenSameHash(a,b):
    length=0
    for i in range(len(a)):
        if a[i]==b[i] and a[i]>0:
            length += 1
    return length


def getBaselineSimilarity(matrix):
    matrix=matrix[:,1:]
    Sim_baseline=numpy.zeros((numpy.size(matrix,0),numpy.size(matrix,0)))
    baseline=[]
    threshold=0.1
    for i in range(numpy.size(matrix,0)):
        sentence1=matrix[i,:]
        for j in range(i+1,numpy.size(matrix,0)):
            sentence2 = matrix[j,:]
            if lenUnion(sentence1,sentence2)>0:
                sim= lenInteraction(sentence1,sentence2)*1.0/(lenUnion(sentence1,sentence2)*1.0)
            else:
                sim =0.0
            Sim_baseline[i,j]=sim
            if sim > threshold:
                print('The baseline similarity between sentence %d and sentence %d is %f' % ((i+1),(j+1),sim))
                baseline.append([(i+1),(j+1)])
    return Sim_baseline

def getMinHashSimilarity(matrix,numHash):
    matrix=matrix[:,1:]
    matrix=matrix[:,:numHash]
    Sim_minhash = numpy.zeros((numpy.size(matrix, 0), numpy.size(matrix, 0)))
    minhash=[]
    threshold = 0.5
    for i in range(numpy.size(matrix,0)):
        sentence1=matrix[i,:]
        for j in range(i + 1, numpy.size(matrix, 0)):
            sentence2 = matrix[j, :]
            sim=lenSameHash(sentence1,sentence2)*1.0/(numHash*1.0)
            Sim_minhash[i, j] = sim
            if sim > threshold:
                print('The baseline similarity between sentence %d and sentence %d is %f' % ((i+1),(j+1),sim))
                minhash.append([(i+1),(j+1)])
    return Sim_minhash

if __name__ == "__main__":
    '''dataset=pd.read_csv('rawresult.csv', sep=',')
    starttime1=time.time()
    Sim_baseline=getBaselineSimilarity(dataset.values)
    df=pd.DataFrame(Sim_baseline)
    df.to_csv('BaselineSimilarity.csv')
    print('Running time for baseline similarity computation is %.2f seconds' % (time.time()-starttime1))'''
    dataset= pd.read_csv('MinHash.csv', sep=',')
    Sim_minhash=getMinHashSimilarity(dataset.values,16)
    df=pd.DataFrame(Sim_minhash)
    df.to_csv('MinHashSimilarity.csv')
