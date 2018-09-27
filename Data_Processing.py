#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:41:52 2018

@author: sameepshah
"""
import pandas as pd

class DataProcessing(object):
    
    
    def read_data(path):
        train =pd.read_csv('titanic_train.csv')
    
        return train
    
  
 
    def imput_age(cols):
        Age = cols[0]
        Pclass = cols[1]
    
        if pd.isnull(Age):
            
            if Pclass ==1:
                return 37
            elif Pclass == 2:
                return 29
            else:
                return 24
        else:
            return Age
    
    
    def data_clean(self,train):
        train['Age'] = train[['Age', 'Pclass']].apply(DataProcessing.imput_age, axis=1)
        train.drop('Cabin', axis=1, inplace=True)
        train.info()
        sex = pd.get_dummies(train['Sex'], drop_first=True)
        embark = pd.get_dummies(train['Embarked'], drop_first=True)
        pclass = pd.get_dummies(train['Pclass'], drop_first=True)
        train = pd.concat([train,sex,embark,pclass], axis=1)
        train.info()
        train.drop(['Sex', 'Embarked', 'Name','Ticket', 'Pclass'], axis =1, inplace=True)
        train.head()
        train.drop('PassengerId', axis = 1, inplace =True)
        #print(train)
        return train

