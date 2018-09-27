#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 23:26:22 2018

@author: sameepshah
"""
import seaborn as sns
import matplotlib.pyplot as plt

def graphs(train):
    fig, axs =plt.subplots(figsize = (10,14),ncols=2, nrows =3)
    #sns.set_style('whitegrid')
    sns.heatmap(train.isnull(),yticklabels=False, cbar=False,cmap='viridis', ax=axs[0,0])
    #sns.set_style('whitegrid')
    sns.countplot(x='Survived',hue='Sex',data=train, palette='RdBu_r', ax=axs[1,0])
    sns.countplot(x='Survived',hue='Pclass',data=train, ax=axs[0,1])
    sns.distplot(train['Age'].dropna(),kde=False,bins=30, ax=axs[1,1])
    sns.countplot(x='SibSp', data=train, ax=axs[2,0])
    #sns.displot(train['Fare'],kde=False,bins=30, ax=axs[2,1])
    train['Fare'].hist(bins=40)



def Heat_map(DF):
    
    sns.heatmap(DF.corr(), annot = True, fmt = ".2f")
    plt.figure('Pearson Correlation of Features')
