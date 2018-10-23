#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xin Jin
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: KNN.py
@time: 10/19/18 7:48 PM
@desc: This is the function to obtain the classification result by K Nearest Neighbor(KNN) method
'''

import numpy as np
from collections import Counter

"*** Determining the neighbors ***"
def computeDistance(instance1, instance2): # Euclidean distance
    return np.linalg.norm(np.array(instance1) - np.array(instance2))

def computeNeighbors(trainingSet, trainingLabels, testInstance, k):
    '''
    description:
        get_neighors calculates a list of the k nearest neighbors of of an instance 'test_instance'.
    :param trainingSet: this should be matrix of the training set
    :param trainingLabels: the list of labels of the training set
    :param testInstance: the test instance
    :param k: the key parameter of k
    :param distance: the method to calculate the distance of obejects from training set and test set
    :return: tuple: (index, dist, label)
    '''
    distances = []
    for i in range(len(trainingSet)):
        dist = computeDistance(testInstance, trainingSet[i])
        distances.append((trainingSet[i], dist, trainingLabels[i]))
    distances.sort(key=lambda x: x[1])
    return distances[:k]

"*** Voting to get a Single Result ***"
def computeLabel(neighbors):
    '''
    description:
        This function is to calculate the label of the current cluster neighbors
    :param neighbors: the neighbors of the current cluster
    :return: the label of current neighbors cluster
    '''
    counter = Counter()
    for i in range(len(neighbors)):
        dist = neighbors[i][1]
        label = neighbors[i][2]
        counter[label] += 1 / (dist**2 + 1)
    winner = counter.most_common(1)[0][0]
    return winner

"*** Classify the test set ***"
def ClassifyTestset(trainingset, testset, trainingLabel, testLabel, K):
    '''
    description:
        this function is to classsify the testset and help user to update better K value
    :param trainingset: training set
    :param testset: test set
    :param trainingLabel: the labels of current training set
    :param testLabel: the labels of current test set
    :param K: the K value for the KNN method
    :return: the classified label for test set
    '''
    ClassifiedLabels=[]
    for i in range(len(testLabel)):
        neighbors = computeNeighbors(trainingset, trainingLabel, testset[i], K)
        classifiedlabel=computeLabel(neighbors)
        ClassifiedLabels.append(classifiedlabel)
    return ClassifiedLabels

"*** Computer the classification accuracy ***"
def ComputeAccuracy(ClassifiedLabels,testLabel):
    '''
    description:
        Compute the accuracy for current test set by comparing the classified labels and the original labels
    :param ClassifiedLabels: the classified labels according to KNN
    :param testLabel: the original label of the test set
    :return: the accuracy for the KNN classification
    '''
    numWrong=0.0
    if len(testLabel)!=len(ClassifiedLabels):
        return 0.0
    else:
        for i in range(len(testLabel)):
            if testLabel[i] != ClassifiedLabels[i]:
                numWrong+=1
        accuracyRate=1.0-numWrong/float(len(testLabel))
    return accuracyRate