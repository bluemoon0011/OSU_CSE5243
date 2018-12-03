#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xin Jin
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: Preprocessing.py
@time: 12/2/18 5:32 PM
@desc: to generate the 3-word shingles as features and represent each word as the matrix, which will be used in the next
       python files.
'''

from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from nltk.stem import SnowballStemmer
import numpy
import csv
import time

def sentence_preprocess(line):
    '''
    @desc:preprocess the sentence: remove characters, numbers; tokenize the sentence; stem the word
    :param line: one line of the raw sentence
    :return: a tokenize words list
    '''
    line = line.strip() # remove the default character '\n', '\r',  '\t',  ' '
    words = WordPunctTokenizer().tokenize(line) # segment words from the sentence
    words = [w.lower() for w in words if w.isalpha()] # remove non-text characters, numbers
    words = [SnowballStemmer('english').stem(w) for w in words] # stem
    # words = [WordNetLemmatizer().lemmatize(w) for w in words]  # stem (the other stem method)
    return words

def get_feature(dataset):
    '''
    @desc: get feature word of the dataset
    :param dataset: the dataset, from which you want to get the feature words
    :return: the feature word list
    '''
    feature=[]
    for line in dataset:
        words=sentence_preprocess(line)
        for w in words:
            #if w not in feature:
            #    feature.append(w) # check whether there is the same word in existing feature lsit
            feature.append(w)  # check whether there is the same word in existing feature lsit
    return feature

def get_shingle(shingle1, K):
    '''
    description:
        to get the K-word shingles vector from current 1-word shingles vector
    :param shingle1: current 1-word shingles
    :param K: number of words in generated shingle vector
    :return: the K-word shingles vector
    '''
    shingle_length=len(shingle1)
    shingle3=[]
    for i in range(shingle_length-K+1):
        shingle3.append(shingle1[i:i+K])
    additionalWord=[]
    for i in range(len(shingle3)-1):
        word1=shingle3[i]
        for j in range(i+1,len(shingle3)):
            word2=shingle3[j]
            if word1[0] in word2:
                if word1[1] in word2:
                    if word1[2] in word2:
                        additionalWord.append(word2)
    for word in additionalWord:
        if word in shingle3:
            shingle3.remove(word)
    return shingle3

def get_shingleVector(shingle,line):
    '''
    @desc: get the feature value vector for each object(sentence); the feature vector is constructed by the frequency of words in the sentence body
    :param feature: feature vector; it's a list of attribute words
    :param line: object; it's the sentence
    :return: feature value vector of each object
    '''
    feature_vector=numpy.zeros(len(shingle),dtype=numpy.int)
    words=sentence_preprocess(line)
    for word in shingle:
        if ((word[0] in words) and (word[1] in words) and (word[2] in words)):
            word_index=shingle.index(word)
            feature_vector[word_index]=1
    return feature_vector

def build_matrix(dataset,shingle,matrix):
    '''
    @desc: build the feature value matrix by merging feature value vector
    :param dataset: sentences dataset
    :param feature_all: a list of feature words
    :param matrix: the feature_value matirx for the first sentence in the dataset
    :return: the merge feature value matrix
    '''
    for line in dataset:
        vector = get_shingleVector(shingle,line)
        matrix=numpy.row_stack((matrix,vector)) # merge the feature value matrix
    return matrix

def removeUselessShingles(shingle3, matrix):
    additionalword=[]
    additionalindex=[]
    for i in range(len(shingle3)):
        if numpy.sum(matrix[:,i])<1:
            additionalword.append(shingle3[i])
            additionalindex.append(i)
    matrix= numpy.delete(matrix,additionalindex,1)
    for w in additionalword:
        shingle3.remove(w)
    return shingle3, matrix

def dataToFrame1(data,columns):# don't set the sentenses as the indexes
    data = list(data)
    file_data = pd.DataFrame(data, columns=columns)
    return file_data

def mergeString(shingle):
    String = []
    for w in shingle:
        String.append(w[0]+' '+w[1]+' '+w[2])
    return String

def test():
    print(shingle1[len(shingle1) - 1])
    print(shingle3[len(shingle3) - 1])
    print(len(shingle1))
    print(len(shingle3))



# Main function
if __name__ == "__main__":
    start = time.time()
    dataset = pd.read_csv('imdb_dataset.csv', sep='\t', names=['sentence', 'label', ]) # read the dataset.csv file and give label for its first and second column
    shingle1 = get_feature(dataset['sentence'])  # get shingles, in which there is only 1 word
    shingle3 = get_shingle(shingle1, 3)
    vector1= get_shingleVector(shingle3, dataset['sentence'][0])
    feature_matirx = build_matrix(dataset['sentence'][1:], shingle3, get_shingleVector(shingle3, dataset['sentence'][0]))  # get value matrix for sentences
    shingle3, matrix=removeUselessShingles(shingle3,feature_matirx)
    print(len(shingle3))
    df = pd.DataFrame(matrix)
    df.to_csv('rawresult.csv')
    df1=pd.DataFrame(shingle3)
    df1.to_csv('shingle.csv')
    Shingle=mergeString(shingle3)
    df2=dataToFrame1(matrix,Shingle)
    df2.to_csv('rawresult1.csv')
    print("Preprocessing time is:", time.time()-start)
