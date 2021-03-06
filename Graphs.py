#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:31:21 2018

@author: sameepshah
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class Graphs(object):

    def plot_data(self, X, xlabel, ylabel):
        plt.scatter(X[:10, 0], X[:10, 1],
                    color='red', marker='x', label='Overweight')
        plt.scatter(X[10:, 0], X[10:, 1],
                    color='green', marker='o', label='Normal')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

    def plot_misclassifications(self, classifier):
        plt.plot(range(1, len(classifier.errors) + 1), classifier.errors,
                 marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Misclassifications')
        plt.show()

    def plot_decision_regions(self, X, y, classifier, xlabel, ylabel, resolution=0.02):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('green', 'red', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=colors[idx],
                        edgecolor='black',
                        marker=markers[idx],
                        label=get_label(cl))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()


def get_label(cl):
    return 'Overweight' if cl == 1 else 'Normal'