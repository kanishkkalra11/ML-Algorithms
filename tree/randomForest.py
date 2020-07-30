from .base import DecisionTree
from sklearn import tree
import pandas as pd
import numpy as np

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

    def Tree(self,features):
        
        Tree = tree.DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, max_features=features)
        
        return Tree        

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        features = 2        
        forest = [self.Tree(features) for i in range(self.n_estimators)]      
        estimators = []
        
        for tree in forest:
#             mylist = list(range(len(X.columns)))
#             sample_index = np.random.choice(mylist, size=features , replace=True, p=None)
#             X_data = None 
#             for j in range(len(sample_index)):
#                 X_data = pd.concat([X_data, X[:, i]] , axis=1,  ignore_index=True).reset_index()            
            estimator = tree
            estimator.fit(X, y)
            estimators.append(estimator)
            self.estimators = estimators
        return

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = []
        
        for i in range(len(self.estimators)):
            predictions.append(self.estimators[i].predict(X))
        
        final_output = []
        for i in range(len(X)):
            output = {}
            for j in range(len(predictions)):
                if predictions[j][i] in output.keys():
                    output[predictions[j][i]] += 1
                else:
                    output[predictions[j][i]] = 1
            
            Class = max(output, key=output.get)
            # print(Class)
            final_output.append(Class)
        final_output = pd.Series(final_output)

        return final_output

   


class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='mse', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        self.n_estimators = n_estimators
        self. criterion = criterion
        self.max_depth = max_depth

    def Tree(self,features):
        
        Tree = tree.DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth, max_features=features)
        
        return Tree

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        features = 2        
        forest = [self.Tree(features) for i in range(self.n_estimators)]      
        estimators = []
        
        for tree in forest:
#             mylist = list(range(len(X.columns)))
#             sample_index = np.random.choice(mylist, size=features , replace=True, p=None)
#             X_data = None 
#             for j in range(len(sample_index)):
#                 X_data = pd.concat([X_data, X[:, i]] , axis=1,  ignore_index=True).reset_index()            
            estimator = tree
            estimator.fit(X, y)
            estimators.append(estimator)
            self.estimators = estimators
        return

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = []
        
        for i in range(len(self.estimators)):
            predictions.append(self.estimators[i].predict(X))
        
        final_output = []
        for i in range(len(X)):
            output = {}
            for j in range(len(predictions)):
                if predictions[j][i] in output.keys():
                    output[predictions[j][i]] += 1
                else:
                    output[predictions[j][i]] = 1
            
            Class = max(output, key=output.get)
            # print(Class)
            final_output.append(Class)
        final_output = pd.Series(final_output)

        return final_output

   
