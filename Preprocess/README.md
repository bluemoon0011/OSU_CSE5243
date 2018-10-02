# OSU CSE5243/Introduction of Data Ming
****
Author: Xin Jin

Email: xin.jin0010@gmail.com
****
# Description
This is for the course "CSE5243/Introduction of Data Mining", the graduate level course of Computer Science and Engineering, The Ohio State Unversity.

There are some projects, including data preprocessing (project 1) and other incoming projects.

# Project 1/Data Preprocessing

## Description

This assignment is the first part of a longer-term project. The objective is to give you the experience of preprocessing real data and preparing it for future tasks such as automated classification.

### Data
Sentiment Labelled Sentences Data Set, which contains sentences labelled with positive or negative sentiment. It can be downloaded here http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences.
Read their readme.txt file for detailed information. There are three sub-sets respectively from IMDB, Amazon and Yelp. Please merge them as asingle dataset, which should contain 3,000 sentences in total.

Each data file is .txt where each row has two columns: sentence body and sentence label. For example, one sample sentence is "Very little music or anything to speak of. 0", where the first column
"Very little music or anything to speak of." is the content of a sentence while the second column "0" is its sentiment label (1 means positive; 0 means negative).


### Task
In this assignment, your task is to construct a feature vector for each sentence in the data set. For now, please use the frequency of words in the sentence body to construct a feature vector. For example, if there are totally M sentences and N words in the dataset, you will construct a
MxN matrix D, where Dij means the count of word j in sentence i. Hint: You first need to segment/tokenize a sentence to get a collection of words in it. After that, it is up to you whether to do stemming (e.g.,"likes" and \liked" are stemmed to \like") or simply keep the original words.

# Project x/Incoming
