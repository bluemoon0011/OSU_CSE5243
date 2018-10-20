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
def distance(instance1, instance2): # Euclidean distance
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1 - instance2)

def get_neighbors(training_set, labels, test_instance, k, distance=distance):
    '''
    desc:
        get_neighors calculates a list of the k nearest neighbors of of an instance 'test_instance'.
    :param training_set: this should be matrix of the training set
    :param labels: the list of labels of the training set
    :param test_instance: the test instance
    :param k: the key parameter of k
    :param distance: the method to calculate the distance of obejects from training set and test set
    :return: tuple: (index, dist, label)
    '''
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

"*** Voting to get a Single Result ***"
def vote(neighbors): # vote the label for one cluster
    '''0.686666666667
    desc:
        try to vote the label of the neighbors which is the label of the test instance
    :param neighbors:
    :return:
    '''
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    return class_counter.most_common(1)[0][0]

def vote_prob(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    return winner, votes4winner/sum(votes)

def vote_harmonic_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
        class_counter[neighbors[index][2]] += 1/(index+1)
    labels, votes = zip(*class_counter.most_common())
    #print(labels, votes)
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)

def vote_distance_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
        dist = neighbors[index][1]
        label = neighbors[index][2]
        class_counter[label] += 1 / (dist**2 + 1)
    labels, votes = zip(*class_counter.most_common())
    #print(labels, votes)
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)

def ClassifyTestset(trainingset, testset,trainingLabel, testLabel,K):
    numTestSample=len(testLabel)
    ClassifiedLabels=[]
    #testSetIndex=testLabel.index.tolist()
    for i in range(len(testLabel)):
        a=testset[i]
        neighbors = get_neighbors(trainingset, trainingLabel, testset[i], K, distance=distance)
        classifiedlabel, voteresult=vote_distance_weights(neighbors,all_results=False)
        #classifiedlabel, voteresult = vote_prob(neighbors)
        ClassifiedLabels.append(classifiedlabel)
    return ClassifiedLabels

def ComputeAccuracy(ClassifiedLabels,testLabel):
    numWrong=0.0
    if len(testLabel)!=len(ClassifiedLabels):
        return 0.0
    else:
        for i in range(len(testLabel)):
            if testLabel[i] != ClassifiedLabels[i]:
                numWrong+=1
        accuracyRate=1.0-numWrong/float(len(testLabel))
    return accuracyRate


"*** Test code ***"
"""
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

np.random.seed(42)
indices = np.random.permutation(len(iris_data))
n_training_samples = 12
learnset_data = iris_data[indices[:-n_training_samples]]
learnset_labels = iris_labels[indices[:-n_training_samples]]
testset_data = iris_data[indices[-n_training_samples:]]
testset_labels = iris_labels[indices[-n_training_samples:]]
labels=ClassifyTestset(learnset_data,testset_data,learnset_labels,testset_labels)
rate=ComputeAccuracy(labels,testset_labels)
print len(learnset_labels)
print len(testset_labels)
print labels
print testset_labels
print distance(labels,testset_labels)
print rate

"""
"""
"*** Import the data set ***"
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target


"*** Split data set ***"
np.random.seed(42)
indices = np.random.permutation(len(iris_data))
n_training_samples = 12
learnset_data = iris_data[indices[:-n_training_samples]]
learnset_labels = iris_labels[indices[:-n_training_samples]]
testset_data = iris_data[indices[-n_training_samples:]]
testset_labels = iris_labels[indices[-n_training_samples:]]

"*** Plot Data Set ***"
colours = ("r", "b")
X = []
for iclass in range(3):
    X.append([[], [], []])
    for i in range(len(learnset_data)):
        if learnset_labels[i] == iclass:
            X[iclass][0].append(learnset_data[i][0])
            X[iclass][1].append(learnset_data[i][1])
            X[iclass][2].append(sum(learnset_data[i][2:]))
colours = ("r", "g", "y")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for iclass in range(3):
       ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
plt.show()


for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data,learnset_labels,testset_data[i],3,distance=distance)
    print("index: ", i,", result of vote: ", vote(neighbors),", label: ", testset_labels[i], ", data: ", testset_data[i])

for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data,
                              learnset_labels,
                              testset_data[i],
                              5,
                              distance=distance)
    print("index: ", i,
          ", vote_prob: ", vote_prob(neighbors),
          ", label: ", testset_labels[i],
          ", data: ", testset_data[i])

for i in range(5):
    neighbors = get_neighbors(learnset_data,learnset_labels,testset_data[i],3,distance=distance)
    print(i,testset_data[i],testset_labels[i],neighbors)
    
for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data,
                              learnset_labels,
                              testset_data[i],
                              6,
                              distance=distance)
    print("index: ", i,
          ", result of vote: ", vote_distance_weights(neighbors,
                                                      all_results=True))
"""