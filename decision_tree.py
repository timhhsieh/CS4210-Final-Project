# ---------------------------------------
# Group Project

# https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.02-Using-Decision-Trees-to-Diagnose


# ---------------------------------------

from sklearn import tree
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn import datasets, model_selection
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()

X = np.array(iris.data)
Y = np.array(iris.target)

print("Before split training testing")
print("Data instances:", X.shape)
print("Target Values: ", Y.shape)

print("Using 20% for testing 80% for training to split data set: ")
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 42)

print("Training sets: ", X_train.shape)
print("Testing sets: ", X_test.shape)


print("tuning hyper-parameters")
max_depths = np.array([1, 2, 3, 5, 7, 9, 11])

train_score = []
test_score = []

for d in max_depths :
    clf = tree.DecisionTreeClassifier(max_depth = d, random_state = 42)
    clf.fit(X_train, y_train)
    train_score.append(clf.score(X_train, y_train))
    test_score.append(clf.score(X_test, y_test))


plt.style.use('ggplot')

plt.figure(figsize = (10, 6))
plt.plot(max_depths, train_score, 'o-', linewidth = 3, label = 'train')
plt.plot(max_depths, test_score, 's-', linewidth = 3, label = 'test')
plt.xlabel('max_depth')
plt.ylabel('score')
plt.ylim(0.85, 1.1)
plt.legend()
plt.show()


#print("Training Performance: ")

#train_pred = banana_tree.predict(X_train)
#t1 = sum(train_pred == y_train)
#print("predict true / # of ground tree ", t1, len(y_test))
#print("--> accuracy = ", t1 / len(y_test))

#print("Test Performance: ")
#test_pred = banana_tree.predict(X_test)
#t2 = sum(test_pred == y_train)
#print("predict true / # of ground tree ", t2, len(y_test))
#print("--> accuracy = ", t2 / len(y_test))
#
#
