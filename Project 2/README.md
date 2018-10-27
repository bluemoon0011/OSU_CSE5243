CSE5243:Introduction to Data Mining
===========================
# 1. About
* Homework #3
* Xin Jin(jin.967@osu.edu)
* python 3.6

# 2. Task Description
## 2.1 Task completion
In the HW #3 assignment, I built and evaluated several classifiers to predict the sentiment
of a given sentence. And I completed this assignment by the following steps.

## 2.2 Dataset split
In this part, I first load the dataset from result of the HW #2 project. I obtain the dataFrame, in which the feature is
the word and its one-hot representation. The original feature vector is FV-1. Then I use the sklearn package to split the 
data set into training set, test set, validation set and the set size is 60%, 20% and 20%.

## 2.3 feature selection
In this part, I first exclude the stop words out of the origianl feature vector. Then I use the TF-IDF value to extract 
the key 1000 feature word vector as FV-2. Then I use the key 1000 feature to prune the training set, test set and validation set.

## 2.4 Classification
In this part, I used several classification algorthms and functions to classify the training set and get the classifiers.
Then I use the validation set to adjust the parameters in the classifiers, such as the K value in the KNN. In this homework, 
I have used 4 classification algorithms. They are K-nearest neighbors (KNN), naive bays, logistic regression and support
vector machine (SVM). 

### 2.4.1 KNN classifiers
I have designed my own KNN algorithm and written into KNN.py file. I also use the classifiers from 
the sklearn package to use its functions. For K-nearest neighbors classifier, I tried it twice. In the first time, I used 
the classifier designed by myself, which is in the `KNN.py` file. In the second time, I used the `KNeighborsClassifier()`
function from the `sklearn` package.

### 2.4.2 Naive bayes classifier
For Naive bayes classifier, I use the `GaussianNB()` function from the `sklearn` package.

### 2.4.3 Logistic regression classifier
For logistic regression classifier, I use the `LogisticRegression()` function from the `sklearn` package.

### 2.4.4 Support vector machine classifier
For support vector machine classifier, I use the `NuSVC()` function function from the `sklearn` package.

# 3. Test code
## 3.1 Download
* Download the code from Carmen.
* Keep `classification.py`, `KNN.py`, `original_dataset.csv` and `result_of_project1.csv` in the same folder.
* Open the terminal and make sure you support python 3.6 and have the package: numpy, sklearn, pandas, re, operator, nltk,
sys, time.

## 3.2 Online classification
* I have trained the model and get the classifier.
* For KNN (my own algorithm), run `python classification.py classify knn1`.
* For KNN (from sklearn package), run `python classification.py classify knn2`.
* For naive bayes, run `python classification.py classify nb`.
* For logistic regression, run `python classification.py classify lr`.
* For support vector machine, run `python classification.py classify svm`.

## 3.3 Offline training
* If you want to re-train the classifiers, you can conduct the following command.
* For KNN (my own algorithm), run `python classification.py train knn1`
* For KNN (from sklearn package), run `python classification.py train knn2`.
* For naive bayes, run `python classification.py train nb`.
* For logistic regression, run `python classification.py train lr`.
* For support vector machine, run `python classification.py train svm`.

# 4. Sources and reference
* Project created using packages:  numpy, sklearn, pandas, re, operator, nltk, sys, time.
* Code developed using the Pycharm IDE.
* Project created citing from [sklearn](http://scikit-learn.org/stable/index.html) as reference for code (was changed by me during implementation to be my own unique code).
* Started to follow [this guide](https://www.python-course.eu/k_nearest_neighbor_classifier.php), but I changed my mind near the beginning and figured out most of it on my own. Code was changed by me during implementation anyways, even when I was following it.
