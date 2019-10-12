from sklearn import datasets
import pandas as pd 
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import matplotlib.pyplot as plt
from scipy import stats
import sys
import math

 
 

#Reading test data and divising in features and labels
testData = pd.read_csv('mnist-in-csv/mnist_test.csv')
print(len(testData))
testY = testData.iloc[:,0]
testX = testData.iloc[:,1:((testData.shape)[1]-1)]


#Reading train data and divising in features and labels
trainData = pd.read_csv('mnist-in-csv/mnist_train.csv')
trainY = trainData.iloc[:,0]
trainX = trainData.iloc[:,1:((trainData.shape)[1]-1)]


#Choice of kernel and SVM characteristics
svc = svm.SVC(kernel='poly', max_iter=-1, gamma='auto')

#Training phase
svc.fit(trainX, trainY)
score_train = svc.score(trainX, trainY)
print("Score d'apprentissage = {:.2f}%".format(score_train*100))
score_test = svc.score(testX, testY)
print("Score de test = {:.2f}%".format(score_test*100))

#Prediction phase
predY = svc.predict(testX)
nequal = (predY == testY).sum()

#Metrics
print("Erreur de test (par predict) = {:.2f}%".format(nequal/testY.shape[0]*100))
c=confusion_matrix(testY, predY)
print("Confusion matrix:")
print(c)




