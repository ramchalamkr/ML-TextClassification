# -*- coding: utf-8 -*-


#Execute this function once input data has been vectorised.

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import operator

TestSplit = 60000



def KNN(XMatrix,toBeClassifiedVector,YVector,K):

    numRows = len(XMatrix[:,0])
    numCols = len(XMatrix[0,:])
    dictionaryDistance = {}    
    listOfDistances = []
    
    for looper in range(numRows):
        distance = XMatrix[looper,:] - toBeClassifiedVector
        distance = distance ** 2 
        #print(distance,np.sum(distance))
        tupleIndexAndDistance = (looper,np.sum(distance))
        listOfDistances.append(tupleIndexAndDistance)
        
    #print(listOfDistances)
    #my_list.sort(key=operator.itemgetter(1))
    listOfDistances.sort(key=operator.itemgetter(1))
    #print(listOfDistances)
    dictionaryOfKClasses = {1:0,2:0,3:0,4:0}
    for i in range(K):
        classLabel = YVector[listOfDistances[i][0],0]
        #print(classLabel)
        dictionaryOfKClasses[classLabel] = dictionaryOfKClasses[classLabel] + 1
    return sorted(dictionaryOfKClasses.items(), key=operator.itemgetter(1),reverse=True)[0][0] 
    #print('The dominating class is ',sorted(dictionaryOfKClasses.items(), key=operator.itemgetter(1),reverse=True)[0][0])

def TestTrainSplit(X,y):
        k = np.random.permutation(X.shape[0])
        train_idx, test_idx = k[:TestSplit], k[TestSplit:]
        X_train,X_test = X[train_idx], X[test_idx]
        y_train,y_test = y[train_idx], y[test_idx]
        return X_train,X_test,y_train,y_test

def PredictionAccuracy(y_pred,y_final):
        return np.mean(y_pred == y_final)



        
x_1 = pickle.load(open('x_train_binary.pkl',"rb"))
y_1 = pickle.load(open('y_train.pkl',"rb"))

#include chisquare-based selection
chi2Estimator = SelectKBest(score_func=chi2, k=2)
x_1_new = chi2Estimator.fit_transform(x_1, y_1)

x_1_new = x_1_new.toarray()


X_train, X_test, y_train, y_test = TestTrainSplit(x_1_new,y_1)
#y_t = y_t.T
y_pred = []

for i in range(len(X_test[:,0])):
    y_pred.append(KNN(X_train,X_test[i],np.reshape(y_train,(len(y_train),1)),3))
print PredictionAccuracy(y_test,y_predfinal)