OSU CSE5243/Introduction of Data Ming
===========================
|Author|Xin Jin|
|---|---
|E-mail|xin.jin0011@gmail.com


****
# Description
This is for the course "CSE5243/Introduction of Data Mining", the graduate level course of Computer Science and Engineering, The Ohio State Unversity.

There are some projects, including data preprocessing (project 1) and other incoming projects.

# Project 1/Data Preprocessing

## Description

This assignment is the first part of a longer-term project. The objective is to give you the experience of preprocessing real data and preparing it for future tasks such as automated classification.

## Data
Sentiment Labelled Sentences Data Set, which contains sentences labelled with positive or negative sentiment. It can be downloaded here http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences.
Read their readme.txt file for detailed information. There are three sub-sets respectively from IMDB, Amazon and Yelp. Please merge them as asingle dataset, which should contain 3,000 sentences in total.

Each data file is .txt where each row has two columns: sentence body and sentence label. For example, one sample sentence is "Very little music or anything to speak of. 0", where the first column
"Very little music or anything to speak of." is the content of a sentence while the second column "0" is its sentiment label (1 means positive; 0 means negative).


## Task
In this assignment, your task is to construct a feature vector for each sentence in the data set. For now, please use the frequency of words in the sentence body to construct a feature vector. For example, if there are totally M sentences and N words in the dataset, you will construct a
MxN matrix D, where Dij means the count of word j in sentence i. Hint: You first need to segment/tokenize a sentence to get a collection of words in it. After that, it is up to you whether to do stemming (e.g.,"likes" and \liked" are stemmed to \like") or simply keep the original words.

# Project 2/Sentiment Classification

You can see the details in the folder Porject 2

# Project 3/Locality Sensitive Hashing
## Description
The objective in this problem is to evaluate the effcacy and effciency of min-wise hashing for document similarity. You will use the Sentiment Labelled Sentence Data Set (i.e., what you used in the previous two programming as-signments). I begin with the feature vector you created in Assignment1 (feel free to adapt or develop alternatives if you wish to start from scratch).
For example some of you used n-grams or 3-word shingles in your feature vector(results on these may yield better results as we discussed in the lecture).

## Dataset
Sentiment Labelled Sentences Data Set, which contains sentences labelled with positive or negative sentiment. It can be downloaded here http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences.
Here I only use the IMDB dataset.

## Task
### Creating a Baseline
For the baseline you use the raw feature vectors and for each pair (of documents/sentences) report the exact Jaccard similarity. We will call this the true similarity baseline.

### Creating a k-minhash sketch
You can use any publicly available tool or create your own minhash sketch for each sentence from its corresponding feature vector following the procedure described in the lecture (Shingling is optional and will net bonus points but does add complexity to the approach). You will then use this k-minhash sketch (setting k at 16 and 128 repectively) and report the estimated Jaccard similarity between every pair of sentences. We will call this k-minhash estimate of similarity.

Your report should compare both strategies along the axes of effciency (time to generate an estimate from sketch) and effcacy for different values of k. For effcacy or quality you may choose to report mean-squared error (between the estimate and the true similarity) or relative mean error (normalized by the true similarity value). You are also expected to plot or graph these ideas in your report to facilitate comparisons across different sketch sizes.

For more details, please read the readme file in Project 3.
