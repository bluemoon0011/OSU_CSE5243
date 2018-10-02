#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xin Jin
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: xin.jin0010@gmail.com
@software: pycharm
@file: preprocess.py
@time: 9/24/18 8:31 PM
@desc:this file is to preprocess the sentiment dataset from amazon, imdb, yelp. To preprocess them, this file collects
    all the words showing up in the sentences into the feature vector, then build up the 0-1 data matrix as the new datasets.
'''

from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from nltk.stem import SnowballStemmer
import numpy
import csv



def TxtToCsv(file_txt,file_csv):
    '''
    @desc: This is a file format converting function; convert file.txt into file.csv
    :param file_txt: file path of file.txt
    :param file_csv: file path of file.csv
    :return: no return
    '''
    with open(file_txt, 'r') as textfile:
        lines=(line.strip() for line in textfile)
        lines=(line.split(",") for line in lines if line)
        with open(file_csv, 'w') as filecsv:
            file=csv.writer(filecsv)
            file.writerows(lines)

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
            if w not in feature:
                feature.append(w) # check whether there is the same word in existing feature lsit
    return feature

def merge_feature(feature1, feature2):
    '''
    @desc: merge two feature lists
    :param feature1: the fist feature list
    :param feature2: the second feature list
    :return: the merged feature list
    '''
    for w in feature2:
        if w not in feature1: # record the different words of feature2 list into feature1 list
            feature1.append(w)
    return feature1

def get_FeatureVector(feature,line):
    '''
    @desc: get the feature value vector for each object(sentence); the feature vector is constructed by the frequency of words in the sentence body
    :param feature: feature vector; it's a list of attribute words
    :param line: object; it's the sentence
    :return: feature value vector of each object
    '''
    feature_vector=numpy.zeros(len(feature),dtype=numpy.int)
    words=sentence_preprocess(line)
    for w in words:
        word_index=feature.index(w)
        feature_vector[word_index]=feature_vector[word_index]+1
    return feature_vector

def build_matrix(dataset,feature_all,matrix):
    '''
    @desc: build the feature value matrix by merging feature value vector
    :param dataset: sentences dataset
    :param feature_all: a list of feature words
    :param matrix: the feature_value matirx
    :return: the merge feature value matrix
    '''
    for line in dataset:
        vector = get_FeatureVector(feature_all,line)
        matrix=numpy.row_stack((matrix,vector)) # merge the feature value matrix
    return matrix

def dataToFrame(data, indexes, columns):
    '''
    @desc: convert the feature value matrix into the dataFrame format
    :param data: feature value matrix
    :param indexes: the index label of dataFrame
    :param columns: the column label of dataFrame
    :return: the final dataFrame of feature value with index and columns
    '''
    data = list(data)
    columns = list(columns)
    indexes=list(indexes)
    file_data = pd.DataFrame(data, index=indexes, columns=columns)
    return file_data

# Main function
if __name__ == "__main__":
    dataset = pd.read_csv('dataset.csv', sep='\t', names=['sentence', 'label', ]) # read the dataset.csv file and give label for its first and second column
    feature = get_feature(dataset['sentence']) # get feature vector
    feature_matirx = build_matrix(dataset['sentence'][1:], feature, get_FeatureVector(feature, dataset['sentence'][0])) #get value matrix for sentences
    df_amazon = dataToFrame(feature_matirx, dataset['sentence'], feature) # convert the value matrix into dataFrame format dateset
    df_amazon.to_csv('result.csv') # write the dataset into result.csv file for next project



