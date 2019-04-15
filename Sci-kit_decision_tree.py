#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 02:47:24 2019

@author: linasaha
"""
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix 

# Load the training data
M = np.genfromtxt('./mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytrn = M[:, 0]
Xtrn = M[:, 1:]

# Load the test data
M = np.genfromtxt('./mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytst = M[:, 0]
Xtst = M[:, 1:]

print('#Bagging...')
for max_depth in [3, 5]:
    model = DecisionTreeClassifier(criterion='entropy',max_depth=max_depth)
    for num_trees in [5, 10]:
        Bagging = BaggingClassifier(base_estimator= model, n_estimators=num_trees).fit(Xtrn, ytrn)
        prediction = Bagging.score(Xtst, ytst)
        print('Accuracy for max_depth =', max_depth, 'and num_trees =', num_trees, 'is:',prediction*100,'%')
        print("Confusion Matrix:")
        print(confusion_matrix(ytst, Bagging.predict(Xtst)))

print('\n')

print('#Boosting...')
for max_depth in [1, 2]:
    model = DecisionTreeClassifier(criterion='entropy',max_depth=max_depth)
    for num_trees in [5, 10]:
        AdaBoost = AdaBoostClassifier(base_estimator= model,n_estimators=num_trees, algorithm='SAMME').fit(Xtrn,ytrn)
        prediction = AdaBoost.score(Xtst, ytst)
        print('Accuracy for max_depth =', max_depth, 'and num_trees =', num_trees, 'is:',prediction*100,'%')
        print('Confusion Matrix:') 
        print(confusion_matrix(ytst, AdaBoost.predict(Xtst)))