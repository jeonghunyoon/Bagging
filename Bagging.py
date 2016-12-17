# -*- coding: utf-8 -*-

__author__ = 'Jeonghun Yoon'

'''
I will implement 'Bagging'. I will use the 'Regression tree' as a base learner.
Output will be a average of results of base learners.
'''

import urllib
import random
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from math import floor
import matplotlib.pyplot as plot

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
treeDepth = 10
modelList = []
predList = []

# Extract samples for bagging
bagProp = 1
nBagSamples = int(len(xTrain) * bagProp)


### 3. Fit models
# Bootstrap Sampling (with replacement)
for iBaseModel in range(nBaseModel):
    sampIdx = []
    for i in range(nBagSamples):
        sampIdx.append(random.choice(range(nTrain)))
    sampIdx = sorted(sampIdx)
    xTrainBag = [xTrain[i] for i in sampIdx]
    yTrainBag = [yTrain[i] for i in sampIdx]

    baseModel = DecisionTreeRegressor(max_depth=treeDepth)
    # Fit a model (Previous models are independant to current fitting.)
    baseModel.fit(xTrainBag, yTrainBag)
    # Predict on test set
    pred = baseModel.predict(xTest)
    predList.append(pred)

    modelList.append(baseModel)


### 4. Assessment for these model.
mse = []
allPredictions = []
nModel = len(modelList)
for iModel in range(nModel):
    prediction = []
    # Prediction : average of models output
    for iPred in range(len(xTest)):
        pred = sum([predList[i][iPred] for i in range(iModel + 1)]) / (iModel + 1)
        prediction.append(pred)
    error = [(yTest[i] - prediction[i]) for i in range(len(yTest))]
    mse.append(sum([e * e for e in error]) / len(error))
    allPredictions.append(prediction)

nModelIdx = [i+1 for i in range(nModel)]


### 5. Plotting
plot.figure()
plot.plot(nModelIdx, mse)
plot.axis("tight")
plot.xlabel("Number of Models in Ensemble")
plot.ylabel("Mean Squared Error")
plot.ylim((0.0, max(mse)))
plot.show()

print ("Minimum MSE")
print(min(mse))