#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:57:11 2018

@author: sameepshah
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

import AdalineSGD
import Data_Processing
import FeatureGraphs
from sklearn.metrics import accuracy_score


def build_plot():
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.tight_layout()
    plt.show()


dp = Data_Processing.DataProcessing()

PATH = "../Adaline/titanic_train.csv"
    
#load the csv file into pandas dataframe
DF = pd.read_csv(PATH)

#Data cleaning
df_train = dp.data_clean(DF)
#df_train = dc.clean_data(DF)

# Determining the most predictive feature
FeatureGraphs.Heat_map(df_train)
    
X1 = df_train.drop('Survived',axis=1).values

min_max_sclaer = preprocessing.MinMaxScaler()
x_scaled = min_max_sclaer.fit_transform(X1)
X = x_scaled
#print(X)

y = df_train['Survived'].values

# Splitting dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state = 11)

# Training Adaline classifier
ada = AdalineSGD.AdalineSGD(eta=0.001, n_iter=100)
ada.fit(X_train, y_train)


# Evaluating on test data
#print(ada.predict(X_test))
acc = accuracy_score(y_test, ada.predict(X_test))
print ("Test Accuracy :: ", acc*100)

# Building plot for the cost function
build_plot()
#task_four()

