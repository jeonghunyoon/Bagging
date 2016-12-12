# -*- coding: utf-8 -*-

__author__ = 'Jeonghun Yoon'

'''
I will implement 'Bagging'. I will use the 'Regression tree' as a base learner.
Output will be a average of results of base learners.
'''

import urllib
import random
from sklearn.cross_validation import train_test_split

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

### 2. Divide data set into train set and test set! Why? avoid overfitting
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=531)
nTrain = len(xTrain)

# Set parameters for ensemble model
nBaseModel = 100
treeDepth = 5
modelList = []
predList = []

# Extract samples for bagging
bagProp = 0.5
nBagSamples = int(len(xTrain) * bagProp)

# Sample index with replacement
for iBaseModel in range(nBaseModel):
    sampIdx = []
    for i in range(nBagSamples):
        sampIdx.append(random.choice(range(nTrain)))
    sampIdx = sorted(sampIdx)
    xTrainBag = [xTrain[i] for i in sampIdx]
    yTrainBag = [yTrain[i] for i in sampIdx]