"""
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
iris = datasets.load_iris()
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print(iris.data)
print(iris.target)
print(gnb.fit(iris.data, iris.target))
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
"""
'''
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
print(type(X),type(y),X.shape,y.shape)
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
print(clf.score(X, y))
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X, y)
print(type(X),type(y))
print(X.shape,y.shape)
print(clf.predict(X))
print(type(clf.predict(X)))

import numpy as np
import time
import sys

if __name__ == "__main__":
    if sys.argv[1]=='classify':
        print(sys.argv[1])
        if sys.argv[2]=='knn':
            print(sys.argv[2])
            a = [1, 0, 1, 0]
            a = np.array(a).astype(int)
            print(np.count_nonzero(a))
            print(a.sum())
            start = time.time()
            for i in range(2000):
                for j in range(2000):
                    pass
            print(time.time()-start)
'''

# for knn1
time1=[61.935927867889404, 61.65877294540405, 61.7736918926239, 62.17240905761719, 61.4989378452301, 61.236366987228394, 61.57503008842468, 2093.7421083450317, 61.82034492492676, 61.78721213340759, 61.55135941505432, 61.615708112716675, 61.97752404212952, 61.81391787528992, 61.70954871177673, 64.14761519432068, 67.2128598690033, 73.94959902763367, 83.0587568283081, 88.65785908699036]
time2=[23.836050987243652, 23.662756204605103, 24.202482223510742, 23.755226135253906, 23.279042959213257, 23.64832091331482, 24.042912006378174, 23.615362882614136, 23.306495904922485, 23.246170043945312, 23.701729774475098, 23.623040914535522, 23.62836003303528, 23.385733127593994, 24.40609312057495, 25.55326199531555, 27.314735174179077, 30.05416178703308, 33.56913900375366, 34.09859585762024]
print(sum(time1),sum(time2))

# for knn2
time1=137.347
time2=33.9143

# for nb
time1=5.3080
time2=1.1558

# for lg
time1=194.2926
time2=40.7801

# for svm
time1=349.4606
time2=88.6658