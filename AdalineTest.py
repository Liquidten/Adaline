#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:57:11 2018

@author: sameepshah
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize

import AdalineSGD
import Data_Processing
import FeatureGraphs


def build_plot():
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.tight_layout()
    plt.show()


def get_accuracy(actual, predicted):
    counter = 0
    for i in range(actual.shape[0]):
        if actual[i] == predicted[i]:
            counter += 1

    accuracy = counter / actual.shape[0] * 100
    print('The accuracy of the prediction is ' + str(round(accuracy, 2)) + '%')



def task_four():
    # Getting features from dataframe
    columns = list(df_train)
    # Adding features and corresponding weights to the list
    weights = []
    for i in range(1, len(ada.w_)):
        weights.append(str(round(abs(ada.w_[i]), 2)) + " " + columns[i])
    # Sorting and printing the features with the highest weight
    print('The most predictive features are:')
    print(sorted(weights, reverse=True)[:3])
    



dp = Data_Processing.DataProcessing()
#df = pd.read_csv('train.csv')
PATH = "../Adaline/titanic_train.csv"
    
#load the csv file into pandas dataframe
DF = pd.read_csv(PATH)

# Determining the most predictive feature
FeatureGraphs.Heat_map(DF)
#FeatureGraphs.graphs(DF)

#Data cleaning
df_train = dp.data_clean(DF)
    
X1 = df_train.drop('Survived',axis=1)
#y = df_train['Survived']

X = X1.values[:, 1:]
X = normalize(X, axis=0, norm='max')
#print(X.shape)

y = df_train.values[:, 0]
#print(y.shape)

# Splitting dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Training Adaline classifier
ada = AdalineSGD.AdalineSGD(eta=0.0001, n_iter=30)
ada.fit(X_train, y_train)

# Evaluating on test data
get_accuracy(y_test, ada.predict(X_test))

# Building plot for the cost function
build_plot()

