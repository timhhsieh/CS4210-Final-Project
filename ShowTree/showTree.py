# ---------------------------------------
# Group Project

# data set: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic


# ---------------------------------------

from sklearn import tree
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from sklearn import datasets, model_selection
from sklearn.tree import DecisionTreeClassifier

data = datasets.load_breast_cancer()

X = data.data
Y = data.target


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 42)


clf = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 1, random_state = 42)
clf.fit(X_train, y_train)


features = ['radius1', 'textures1', 'perimeter1', 'area1', 'smoothness1', 'compactness1',
            'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1',
            'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2',
            'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2',
            'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3',
            'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']

plt.figure(figsize = (12, 12))
tree.plot_tree(clf, feature_names = features, fontsize=5)
plt.show()
