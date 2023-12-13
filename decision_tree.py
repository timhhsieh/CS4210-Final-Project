# ---------------------------------------
# Group Project

# https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.02-Using-Decision-Trees-to-Diagnose


# ---------------------------------------
from sklearn import datasets
import sklearn.model_selection as ms
from sklearn import tree
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import csv

def main():

    X_full = []
    y_encoded = []

    with open("wdbc.csv", mode='r') as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            y_encoded.append(np.where(lines == 'M', 1, 0))


    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_encoded)

    X_train, X_test, y_train, y_test = ms.train_test_split(X_full, y_encoded, stratify = y, test_size = 0.2,
                                                           random_state = 42)

    clf = tree.DecisionTreeClassifier(random_state = 42)

    clf.fit(X_train, y_train)

    clf.score(X_test, y_test)

    # with open("tree.dot", 'w') as f :
    #    f = tree.export_graphviz(dtc, out_file = f, feature_names = data.feature_names, class_names =
    #    data.target_names)

    # tuning hyper-parameters
    max_depths = np.array([1, 2, 3, 5, 7, 9, 11])

    train_score = []
    test_score = []
    for d in max_depths :
        clf = tree.DecisionTreeClassifier(max_depth = d, random_state = 42)
        clf.fit(X_train, y_train)
        train_score.append(clf.score(X_train, y_train))
        test_score.append(clf.score(X_test, y_test))

    #plt.style.use('ggplot')

    #plt.figure(figsize = (10, 6))
    #plt.plot(max_depths, train_score, 'o-', linewidth = 3, label = 'train')
    #plt.plot(max_depths, test_score, 's-', linewidth = 3, label = 'test')
    #plt.xlabel('max_depth')
    #plt.ylabel('score')
    #plt.ylim(0.85, 1.1)
    #plt.legend()
