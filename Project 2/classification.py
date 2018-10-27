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
from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
import pandas as pd
import re
import operator
from nltk.corpus import stopwords
import KNN as knn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC,SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import sys
import time

def stringTofloat(list):
    '''
    description:
        This function is to convert the character list element into float format numbers
    :param list: the list need to be processed
    :return: the list of int numbers
    '''
    for i in range(len(list)): # check all the elements in the list
        l=list[i] # get the list element
        l = re.sub("\D", "", l)# remove the un-digit characters, "\D" matches anything other than character marked as digits in the Unicode character properties database
        l=float(l)
        list[i]=l
    return list

def splitDataset(attributeset,labelset,splitsize):
    '''
    description:
        to split the whole date set into the two data sets(maybe training set and test set)
    :param attributeset: the features value of the instances in the data set, it should be a matrix of numbers
    :param labelset: the labels of the data set
    :param splitsize: the rate of splitting, which should be the size(attributeset_test)/size(attributeset)
    :return: two split attribute sets and two split label sets
    '''
    attributeset_train, attributeset_test, labelset_train, labelset_test = train_test_split(attributeset,labelset,test_size=splitsize)
    return attributeset_train,attributeset_test,labelset_train,labelset_test

def computeTF_IDF(trainingSet,feature):
    '''
    description:
        compute the tf-idf value of each feature in the dataset and then collect all the tf-idf into a list
    :param trainingSet: train set
    :param feature: the feature list of the trainingSet
    :return: the list of tf-idf value
    '''
    tfScore=np.zeros(len(feature),dtype=float) # initlize the list of tf-idf score as the all-zero list
    numInstance,useless=trainingSet.values.shape # get the number of instances
    for i in range(len(feature)):# check all the features
        value=trainingSet[feature[i]].values.astype(int) # get the value of the i-th feature
        tf=sum(value) # get the frequency of the feature
        numContainFeature=np.count_nonzero(value)
        idf=np.log2((numInstance+1.0)/(numContainFeature+1.0))# calculate the idf value
        tfScore[i]=tf*idf #get the tf-idf score
    return list(tfScore)

def chooseKeyFeature(tfScore,feature,N): # this is used to choose N features with the larger tfScore
    '''
    description:
        this function is to get the N features from the given tf-idf score list
    :param tfScore: a list of tf-idf scores
    :param feature: a list of features
    :param N: the most importance N features
    :return: the chosen N features
    '''
    dictionary = dict(zip(feature, tfScore))
    sorted_list= sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True) #make the feature and it tf-idf score as a dictionary, then sort the dictionary
    keyFeature=list()
    for i in range(N):
        keyFeature.append(sorted_list[i][0])
    return keyFeature

def getFeatures(originalDataset,N):
    '''
    description:
        get the feature vectors, including the original feature vector FV-1 and key feature vector FV-2
    :param originalDataset: the original data set from the last project
    :param N: the key N features
    :return: FV-1, FV-2
    '''
    feature=originalDataset.columns.values.tolist()
    feature = feature[1:]
    filteredFeature = [word for word in feature if word not in stopwords.words('english')]
    keyFeature = chooseKeyFeature(computeTF_IDF(originalDataset, filteredFeature), filteredFeature, N)
    return feature,keyFeature

def myKnnClassification(dataset1,dataset2,labelset1,labelset2,FV1,FV2,K):
    '''
    description:
        this is a classifier using K-nearest neighbor (KNN) classification to classify the data set, I designed my KNN
        algorithm and wrote it in to KNN.py file
    :return: the accuracy of FV-1 dataset, the accuracy of FV-2 dataset
    '''
    # test the accuracy rate of data set with FV-1 features
    starttime1=time.time()# record start time
    trainingSet_unpruned=dataset1[FV1]
    testSet_unpruned=dataset2[FV1]
    trainingLabel=labelset1
    testLabel=labelset2
    ClassifiedLabels1 = knn.ClassifyTestset(trainingSet_unpruned.values, testSet_unpruned.values, trainingLabel.values,testLabel.values, K)# the classified labels for the test data set
    accurracy1 = knn.ComputeAccuracy(ClassifiedLabels1, testLabel.values)# compare the classified labels with the test set labels and calculate the accuracy
    time1=time.time()-starttime1 # the time for classify data set with FV-1

    # test the accuracy rate of data set with FV-2 features
    starttime2 = time.time()  # record start time
    trainingSet_pruned=dataset1[FV2]
    testSet_pruned=dataset2[FV2]
    ClassifiedLabels2 = knn.ClassifyTestset(trainingSet_pruned.values, testSet_pruned.values, trainingLabel.values,testLabel.values, K)
    accurracy2 = knn.ComputeAccuracy(ClassifiedLabels2, testLabel.values)
    time2 = time.time() - starttime2  # the time for classify data set with FV-2
    return accurracy1,accurracy2, time1, time2 # get accuracy and time

def knnClassification(dataset1,dataset2,labelset1,labelset2,FV1,FV2,K):
    '''
    description:
        this is a classifier using K-nearest neighbor (KNN) classification to classify the data set, I used the sklearn
        package and used its knn function as the classifier
    :return:
    '''
    # test the accuracy rate of data set with FV-1 features
    starttime1 = time.time()  # record start time
    trainingSet_unpruned = dataset1[FV1]
    testSet_unpruned = dataset2[FV1]
    trainingLabel = labelset1
    testLabel = labelset2
    Knn1 = KNeighborsClassifier(n_neighbors=K)#get KNeighborsClassifier function from the sklearn package
    Knn1.fit(trainingSet_unpruned.values.astype(int), trainingLabel.values.astype(int))# fit the training set with its labels
    ClassifiedLabels1=Knn1.predict(testSet_unpruned.values.astype(int))#predict the test set and get the classified labels
    accurateprediction1 = (ClassifiedLabels1 == testLabel.values.astype(int)).sum()# calculate the number of correctly classified labels
    accurracy1 = accurateprediction1 / len(ClassifiedLabels1)# get the accuracy
    time1 = time.time() - starttime1  # the time for classify data set with FV-1

    # test the accuracy rate of data set with FV-2 features
    starttime2 = time.time()  # record start time
    trainingSet_pruned = dataset1[FV2]
    testSet_pruned = dataset2[FV2]
    Knn2 = KNeighborsClassifier(n_neighbors=K)#get KNeighborsClassifier function from the sklearn package
    Knn2.fit(trainingSet_pruned.values.astype(int), trainingLabel.values.astype(int))# fit the training set with its labels
    ClassifiedLabels2 = Knn2.predict(testSet_pruned.values.astype(int))#predict the test set and get the classified labels
    accurateprediction2 = (ClassifiedLabels2 == testLabel.values.astype(int)).sum()# calculate the number of correctly classified labels
    accurracy2 = accurateprediction2 / len(ClassifiedLabels2)# calculate the number of correctly classified labels
    time2 = time.time() - starttime2  # the time for classify data set with FV-2
    return accurracy1,accurracy2,time1,time2# get the accuracy and time

def naiveBayesClassification(dataset1,dataset2,labelset1,labelset2,FV1,FV2,naivebayes):
    '''
    description:
       this is a classifier using naive bayes classification to classify the data set, I used the sklearn package and
       used its naive bayes function as the classifier
    :return:
    '''
    # test the accuracy rate of data set with FV-1 features
    starttime1 = time.time()  # record start time
    trainingSet_unpruned = dataset1[FV1]
    testSet_unpruned = dataset2[FV1]
    trainingLabel = labelset1
    testLabel = labelset2
    NaiveBayes1 = naivebayes()#get GaussianNB function from the sklearn package
    NaiveBayes1.fit(trainingSet_unpruned.values.astype(int), trainingLabel.values.astype(int))# fit the training set with its labels
    ClassifiedLabels1 = NaiveBayes1.predict(testSet_unpruned.values.astype(int))#predict the test set and get the classified labels
    accurateprediction1 = (ClassifiedLabels1 == testLabel.values.astype(int)).sum()# calculate the number of correctly classified labels
    accurracy1 = accurateprediction1 / len(ClassifiedLabels1)# get the accuracy
    time1 = time.time() - starttime1  # the time for classify data set with FV-1

    # test the accuracy rate of data set with FV-2 features
    starttime2 = time.time()  # record start time
    NaiveBayes2 = naivebayes()
    trainingSet_pruned = dataset1[FV2]
    testSet_pruned = dataset2[FV2]
    NaiveBayes2.fit(trainingSet_pruned.values.astype(int), trainingLabel.values.astype(int))
    ClassifiedLabels2 = NaiveBayes2.predict(testSet_pruned.values.astype(int))
    accurateprediction2 = (ClassifiedLabels2 == testLabel.values.astype(int)).sum()
    accurracy2 = accurateprediction2 / len(ClassifiedLabels2)
    time2 = time.time() - starttime2  # the time for classify data set with FV-2
    return accurracy1, accurracy2, time1, time2  # get the accuracy and time

def logisticRegressionClassification(dataset1,dataset2,labelset1,labelset2,FV1,FV2,solution):
    '''
    description:
        this is a classifier using logistic regression classification to classify the data set, I used the sklearn package
        and used its logistic regression function as the classifier
    :return:
    '''
    # test the accuracy rate of data set with FV-1 features
    starttime1 = time.time()  # record start time
    trainingSet_unpruned = dataset1[FV1]
    testSet_unpruned = dataset2[FV1]
    trainingLabel = labelset1
    testLabel = labelset2
    classifier1 = LogisticRegression(random_state=0, solver=solution, multi_class='multinomial').fit(trainingSet_unpruned.values.astype(int), trainingLabel.values.astype(int))#build the classifier
    ClassifiedLabels1 = classifier1.predict(testSet_unpruned.values.astype(int))#predict the test set and get the classified labels
    accurateprediction1 = (ClassifiedLabels1== testLabel.values.astype(int)).sum()# calculate the number of correctly classified labels
    accurracy1 = accurateprediction1 / len(ClassifiedLabels1)# get the accuracy
    time1 = time.time() - starttime1  # the time for classify data set with FV-1

    # test the accuracy rate of data set with FV-2 features
    starttime2 = time.time()  # record start time
    trainingSet_pruned = dataset1[FV2]
    testSet_pruned = dataset2[FV2]
    classifier2 = LogisticRegression(random_state=0, solver=solution, multi_class='multinomial').fit(trainingSet_pruned.values.astype(int), trainingLabel.values.astype(int))
    ClassifiedLabels2 = classifier2.predict(testSet_pruned.values.astype(int))
    accurateprediction2 = (ClassifiedLabels2 == testLabel.values.astype(int)).sum()
    accurracy2 = accurateprediction2 / len(ClassifiedLabels2)
    time2 = time.time() - starttime2  # the time for classify data set with FV-2
    return accurracy1, accurracy2, time1, time2  # get the accuracy and time

def supportVectorMachineClassification(dataset1,dataset2,labelset1,labelset2,FV1,FV2,svm):
    '''
    description:
        this is a classifier using support vector machine classification to classify the data set, I used the sklearn package
        and used its support vector machine function as the classifier
    :return:
    '''
    # test the accuracy rate of data set with FV-1 features
    starttime1 = time.time()  # record start time
    trainingSet_unpruned = dataset1[FV1]
    testSet_unpruned = dataset2[FV1]
    trainingLabel = labelset1
    testLabel = labelset2
    if svm==LinearSVC:
        classifier1 = svm()# get NuSVC function from the sklearn package
        classifier2 = svm()# get NuSVC function from the sklearn package
    else:
        classifier1 = svm(gamma='auto')  # get NuSVC function from the sklearn package
        classifier2 = svm(gamma='auto')
    #classifier1 = SVC(gamma='auto')#get NuSVC function from the sklearn package
    classifier1.fit(trainingSet_unpruned.values.astype(int), trainingLabel.values.astype(int))#fit the training set with its labels
    ClassifiedLabels1 = classifier1.predict(testSet_unpruned.values.astype(int))#predict the test set and get the classified labels
    accurateprediction1 = (ClassifiedLabels1 == testLabel.values.astype(int)).sum()# calculate the number of correctly classified labels
    accurracy1 = accurateprediction1 / len(ClassifiedLabels1)# get the accuracy
    time1 = time.time() - starttime1  # the time for classify data set with FV-1

    # test the accuracy rate of data set with FV-2 features
    starttime2 = time.time()  # record start time
    trainingSet_pruned = dataset1[FV2]
    testSet_pruned = dataset2[FV2]
    classifier2.fit(trainingSet_pruned.values.astype(int), trainingLabel.values.astype(int))
    ClassifiedLabels2 = classifier2.predict(testSet_pruned.values.astype(int))
    accurateprediction2 = (ClassifiedLabels2 == testLabel.values.astype(int)).sum()
    accurracy2 = accurateprediction2 / len(ClassifiedLabels2)
    time2 = time.time() - starttime2  # the time for classify data set with FV-2
    return accurracy1, accurracy2, time1, time2  # get the accuracy and time


if __name__ == "__main__":
    "*** Data preprocessing ***"
    dataset = pd.read_csv('original_dataset.csv', sep='\t', names=['sentence', 'label', ]) # read the origianl text dataset which contains 3000 sentences and labels
    label = stringTofloat(dataset['label'])# the list of labels
    resultset = pd.read_csv('result_of_project1.csv', sep=',')# read the dataset, which is the resulf of Project 1
    trainingSet, testSet0, trainingLabel, testLabel0 = splitDataset(resultset, label, 0.4)  # get 60% as test set/label, 40% as raw test set/label
    testSet, validationSet, testLabel, validationLabel = splitDataset(testSet0, testLabel0, 0.5)  # split the raw test set as test and validation set, and the test and validation set size is 20% and 20%
    FV1,FV2 = getFeatures(resultset, 1000) # use TF-IDF value to extract 1000 key feature words
    trainingSet_unpruned = trainingSet[FV1]# get the training data set with feature vector FV-1
    testSet_unpruned = testSet[FV1]# get the test data set with feature vector FV-1
    trainingSet_pruned=trainingSet[FV2]# get the training data set with feature vector FV-2
    testSet_pruned=testSet[FV2]# get the test data set with feature vector FV-2

    if sys.argv[1]=='classify':
        if sys.argv[2]=='knn1':
            #"*** Classification by KNN (my own KNN algorithm)***"
            print("*** Classification by KNN (my own KNN algorithm)***")
            accurracy1, accurracy2,time1,time2 = myKnnClassification(trainingSet,testSet,trainingLabel,testLabel,FV1,FV2, 8)  # get the classifiction accuracy for datasets with FV-1 and FV-2
            print("For K-nearest neighbors classifier: the accuracy with FV-1 is %.4f, the accuracy with FV-2 is %.4f." % (accurracy1, accurracy2))
            print("For K-nearest neighbors classifier: the running time with FV-1 is %.4f, the running time with FV-2 is %.4f." % (time1,time2))

        elif sys.argv[2]=='knn2':
            #"*** Classification by KNN (KNN algorithm from sklearn package) ***"
            print("*** Classification by KNN (KNN algorithm from sklearn package) ***")
            accurracy1, accurracy2,time1,time2 = knnClassification(trainingSet,testSet,trainingLabel,testLabel,FV1,FV2, 11)
            print("For K-nearest neighbors classifier: the accuracy with FV-1 is %.4f, the accuracy with FV-2 is %.4f." % (accurracy1, accurracy2))
            print("For K-nearest neighbors classifier, the running time with FV-1 is %.4f, the running time with FV-2 is %.4f." % (time1, time2))

        elif sys.argv[2]=='nb':
            #"*** Classification by Naive Bayes from sklearn package***"
            print("*** Classification by Naive Bayes from sklearn package***")
            accurracy1,accurracy2,time1,time2=naiveBayesClassification(trainingSet,testSet,trainingLabel,testLabel,FV1,FV2, MultinomialNB) # MultinomialNB, BernoulliNB, and GaussianNB
            print("For naive bayes classifier: the accuracy with FV-1 is %f, the accuracy with FV-2 is %f" % (accurracy1,accurracy2))
            print("For naive bayes classifier: the running time with FV-1 is %.4f, the running time with FV-2 is %.4f." % (time1, time2))

        elif sys.argv[2] == 'lr':
            #"*** Classification by logistic regression from sklearn package***"
            print("*** Classification by logistic regression from sklearn package***")
            accurracy1,accurracy2,time1,time2=logisticRegressionClassification(trainingSet,testSet,trainingLabel,testLabel,FV1,FV2,'lbfgs') #‘newton-cg’, ‘lbfgs’, ‘sag’, ‘saga’
            print("For logistic regression classifier: the accuracy with FV-1 is %f, the accuracy with FV-2 is %f" % (accurracy1,accurracy2))
            print("For logistic regression classifier: the running time with FV-1 is %.4f, the running time with FV-2 is %.4f." % (time1, time2))

        elif sys.argv[2] == 'svm':
            #"*** Classification by support vector machine from sklearn package***"
            print("*** Classification by support vector machine from sklearn package***")
            accurracy1,accurracy2,time1,time2=supportVectorMachineClassification(trainingSet,testSet,trainingLabel,testLabel,FV1,FV2, NuSVC) # SVC,LinearSVC,NuSVC
            print("For support vector machine classifier, the accuracy for test set with FV-1 is %f, the accuracy for test set with FV-2 is %f" % (accurracy1,accurracy2))
            print("For support vector machine classifier: the running time with FV-1 is %.4f, the running time with FV-2 is %.4f." % (time1, time2))

    elif sys.argv[1]=='train':
        if sys.argv[2]=='knn1':
            #"*** Training by KNN (my own KNN algorithm)***"
            print("*** Training by KNN (my own KNN algorithm)***")
            starttime=time.time()
            Accuracy1 = []
            Accuracy2 = []
            Time1 = []
            Time2 = []
            for k in range(2,62,3):
                print("Training, K=", k)
                accurracy1, accurracy2, time1, time2 = myKnnClassification(trainingSet, validationSet, trainingLabel,validationLabel, FV1, FV2,k)  # get the classifiction accuracy for datasets with FV-1 and FV-2
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)
            print("For K-nearest neighbors classifier: when train model by changing K value in list [2,5,8,..,62], the accuracy list with FV-1 is ", Accuracy1)
            print("For K-nearest neighbors classifier: when train model by changing K value in list [2,5,8,..,62], the running time list with FV-1 is ", Time1)
            print("For K-nearest neighbors classifier: when train model by changing K value in list [2,5,8,..,62], the accuracy list with FV-2 is ", Accuracy2)
            print("For K-nearest neighbors classifier: when train model by changing K value in list [2,5,8,..,62], the running time list with FV-2 is ", Time2)
            print("For K-nearest neighbors classifier: the total training time is %.4f", time.time()-starttime)

        if sys.argv[2] == 'knn2':
            #"*** Training by KNN (KNN algorithm from sklearn package)***"
            print("*** Training by KNN (KNN algorithm from sklearn package)***")
            starttime = time.time()
            Accuracy1 = []
            Accuracy2 = []
            Time1 = []
            Time2 = []
            for k in range(2, 62, 3):
                print("Training, K=",k)
                accurracy1, accurracy2, time1, time2 = knnClassification(trainingSet, validationSet, trainingLabel,validationLabel, FV1, FV2, k)  # get the classifiction accuracy for datasets with FV-1 and FV-2
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)
            print("For K-nearest neighbors classifier: when train model by changing K value in list [2,5,8,..,62], the accuracy list with FV-1 is ", Accuracy1)
            #print("For K-nearest neighbors classifier: when train model by changing K value in list [2,5,8,..,62], the running time list with FV-1 is ", Time1)
            print("For K-nearest neighbors classifier: when train model by changing K value in list [2,5,8,..,62], the accuracy list with FV-2 is ", Accuracy2)
            #print("For K-nearest neighbors classifier: when train model by changing K value in list [2,5,8,..,62], the running time list with FV-2 is ", Time2)
            print("For K-nearest neighbors classifier: the total training time with FV-1 is %.4f, the total training time with FV-2 is %.4f " % (sum(Time1),sum(Time2)))
        if sys.argv[2] == 'nb':
            #"*** Training by Naive Bayes from sklearn package***"
            print("*** Training by naive bayes from sklearn package***")
            Accuracy1 = []
            Accuracy2 = []
            Time1 = []
            Time2 = []
            starttime = time.time()
            for i in range(10):
                print("Traing, round=",i)
                trainingSet, testSet0, trainingLabel, testLabel0 = splitDataset(resultset, label,0.4)  # get 60% as test set/label, 40% as raw test set/label
                testSet, validationSet, testLabel, validationLabel = splitDataset(testSet0, testLabel0,0.5)  # split the raw test set as test and validation set, and the test and validation set size is 20% and 20%
                FV1, FV2 = getFeatures(resultset, 1000)  # use TF-IDF value to extract 1000 key feature words
                "*** Try BernoulliNB ***"
                accurracy1, accurracy2, time1, time2 = naiveBayesClassification(trainingSet,validationSet,trainingLabel,validationLabel,FV1,FV2, BernoulliNB)
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)
                "*** Try MultinomialNB ***"
                accurracy1, accurracy2, time1, time2 = naiveBayesClassification(trainingSet,validationSet,trainingLabel,validationLabel,FV1,FV2, MultinomialNB)
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)
                "*** Try GaussianNB ***"
                accurracy1, accurracy2, time1, time2 = naiveBayesClassification(trainingSet,validationSet,trainingLabel,validationLabel,FV1,FV2, GaussianNB)
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)
            print("For naive bayes classifier: when train model by changing naive bayes function in list [BernoulliNB, MultinomialNB, GaussianNB], the accuracy list with FV-1 is " % Accuracy1)
            print("For naive bayes classifier: when train model by changing naive bayes function in list [BernoulliNB, MultinomialNB, GaussianNB], the accuracy list with FV-2 is " % Accuracy2)
            print("For K-nearest neighbors classifier: the total training time with FV-1 is %.4f, the total training time with FV-2 is %.4f " % (sum(Time1), sum(Time2)))

        if sys.argv[2] == 'lr':
            # "*** Training by logistic regression from sklearn package***"
            print("*** Training by logistic regression from sklearn package***")
            starttime = time.time()
            Accuracy1 = []
            Accuracy2 = []
            Time1 = []
            Time2 = []
            for i in range(10):
                print("Traing, round=", i)
                trainingSet, testSet0, trainingLabel, testLabel0 = splitDataset(resultset, label,0.4)  # get 60% as test set/label, 40% as raw test set/label
                testSet, validationSet, testLabel, validationLabel = splitDataset(testSet0, testLabel0,0.5)  # split the raw test set as test and validation set, and the test and validation set size is 20% and 20%
                FV1, FV2 = getFeatures(resultset, 1000)  # use TF-IDF value to extract 1000 key feature words

                "*** Try newton-cg ***"
                accurracy1, accurracy2, time1, time2=logisticRegressionClassification(trainingSet, validationSet, trainingLabel, validationLabel, FV1, FV2, 'newton-cg')
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)

                "*** Try lbfgs ***"
                accurracy1, accurracy2, time1, time2=logisticRegressionClassification(trainingSet, validationSet, trainingLabel, validationLabel, FV1, FV2, 'lbfgs')
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)

                "*** Try sag ***"
                accurracy1, accurracy2, time1, time2=logisticRegressionClassification(trainingSet, validationSet, trainingLabel, validationLabel, FV1, FV2, 'sag')
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)
                "*** Try saga ***"
                accurracy1, accurracy2, time1, time2 = logisticRegressionClassification(trainingSet, validationSet, trainingLabel, validationLabel, FV1, FV2, 'saga')
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)
            print("For logistic regression classifier: when train model by changing logistic regression solver in list [newton-cg, lbfgs, sag, saga], the accuracy list with FV-1 is " % Accuracy1)
            print("For logistic regression classifier: when train model by changing logistic regression solver in list [newton-cg, lbfgs, sag, saga], the accuracy list with FV-2 is " % Accuracy2)
            print("For logistic regression classifier: the total training time with FV-1 is %.4f, the total training time with FV-2 is %.4f " % (sum(Time1), sum(Time2)))

        if sys.argv[2] == 'svm':
            # "*** Training by support vector machine from sklearn package***"
            print("*** Training by support vector machine from sklearn package***")
            starttime=time.time()
            Accuracy1 = []
            Accuracy2 = []
            Time1 = []
            Time2 = []
            for i in range(10):
                print("Traing, round=", i)
                trainingSet, testSet0, trainingLabel, testLabel0 = splitDataset(resultset, label, 0.4)  # get 60% as test set/label, 40% as raw test set/label
                testSet, validationSet, testLabel, validationLabel = splitDataset(testSet0, testLabel0, 0.5)  # split the raw test set as test and validation set, and the test and validation set size is 20% and 20%
                FV1, FV2 = getFeatures(resultset, 1000)  # use TF-IDF value to extract 1000 key feature words

                "*** Try SVC ***"
                accurracy1, accurracy2, time1, time2 = supportVectorMachineClassification(trainingSet, validationSet,trainingLabel, validationLabel, FV1, FV2, SVC)  # SVC,LinearSVC,NuSVC
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)

                "*** Try LinearSVC ***"
                accurracy1, accurracy2, time1, time2 = supportVectorMachineClassification(trainingSet, validationSet,trainingLabel, validationLabel, FV1, FV2, LinearSVC)  # SVC,LinearSVC,NuSVC
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)

                "*** Try NuSVC ***"
                accurracy1, accurracy2, time1, time2 = supportVectorMachineClassification(trainingSet, validationSet,trainingLabel, validationLabel, FV1, FV2, NuSVC)  # SVC,LinearSVC,NuSVC
                Accuracy1.append(accurracy1)
                Accuracy2.append(accurracy2)
                Time1.append(time1)
                Time2.append(time2)
            print("For support vector machine classifier: when train model by changing support vector machine function in list [SVC, LinearSVC, NuSVC], the accuracy list with FV-1 is " % Accuracy1)
            print("For support vector machine classifier: when train model by changing support vector machine function in list [SVC, LinearSVC, NuSVC], the accuracy list with FV-2 is " % Accuracy2)
            print("For support vector machine classifier: the total training time with FV-1 is %.4f, the total training time with FV-2 is %.4f " % (sum(Time1), sum(Time2)))