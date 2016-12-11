# -*- coding: utf-8 -*-

__author__ = 'Jeonghun Yoon'

'''
I will implement 'Bagging'. I will use the 'Regression tree' as a base learner.
Output will be a average of results of base learners.
'''

import urllib

### 1. Load data from UCI repository and
xData = []
yData = []
f = urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")

# Split titles and datas
lines = f.readlines()
titles = lines[:1]
lines = lines[1:]

for line in lines:
    tokens = line.strip().split(';')
    # Extract target value
    yData.append(float(tokens[-1]))
    del(tokens[-1])
    # Extract data
    xData.append(map(float, tokens))

nData = len(xData)
nFeat = len(xData[0])

### 2. Divide data set into train set and test set


