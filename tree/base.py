"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index, gini_gain

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion, max_depth):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.best_feature = 0
        self.split_point = 0
        self.lt = 0
        self.gt = 0

    def fit(self, X, y,sample_weights):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        features = list(X.columns)
        gains = []
        split_points = []
        split_indices = []
        if(self.criterion == "information_gain"):
            for i in range(len(features)):
                a,b,c = information_gain(y,X[features[i]],sample_weights)
                gains.append(a)
                split_points.append(b)
                split_indices.append(c)
            ind = np.argmax(gains)
            best_feature = features[ind]
            split_point = split_points[ind]
            split_index = split_indices[ind]
        elif(self.criterion == "gini_index"):
            for i in range(len(features)):
                a,b,c = gini_gain(y,X[features[i]],sample_weights)
                gains.append(a)
                split_points.append(b)
                split_indices.append(c)
            ind = np.argmin(gains)
            best_feature = features[ind]
            split_point = split_points[ind]
            split_index = split_indices[ind]
        x = list(X[best_feature])
        Y = list(y)
        x,Y = (list(t) for t in zip(*sorted(zip(x, Y))))
        y1 = Y[:split_index+1]
        classes, counts = np.unique(y1, return_counts=True)
        h = np.argmax(counts)
        lt = classes[h]
        y2 = Y[split_index+1:]
        classes, counts = np.unique(y2, return_counts=True)
        h = np.argmax(counts)
        gt = classes[h]
        self.best_feature = best_feature
        self.split_point = split_point
        self.lt = lt
        self.gt = gt
        return


    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        x = list(X[self.best_feature])
        y = []
        for i in range(len(x)):
            if (x[i]<self.split_point):
                y.append(self.lt)
            else:
                y.append(self.gt)
        return pd.Series(y)


    
