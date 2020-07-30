import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []
        self.X = 0
        self.y = 0

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        N = len(y)
        sample_weights = np.ones(N)/N
        alphas = []
        estimators = []
        
        for i in range(self.n_estimators):
            estimator = self.base_estimator
            estimator.fit(X,y,sample_weights)
            # estimator.plot()
            estimators.append(estimator)
            y_predict = estimator.predict(X)
            err = 0
            for j in range(N):
                if y[j]!=y_predict[j]:
                    err += sample_weights[j]
            alpha = 0.5 * np.log((1. - err)/err)
            alphas.append(alpha)
            sample_weights = list(sample_weights)
            for j in range(N):
                if y[j]!=y_predict[j]:
                    sample_weights[j] *= np.exp(alpha)
                else:
                    sample_weights[j] *= np.exp(-alpha)
            sample_weights = np.array(sample_weights)
            sample_weights = sample_weights/np.sum(sample_weights)

        self.estimators = estimators
        self.alphas = alphas
        return
            

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = []
        final_preds = []
        for i in range(self.n_estimators):
            predictions.append(self.estimators[i].predict(X))
        for i in range(len(predictions[0])):
            classes = {}
            for j in range(len(predictions)):
                try:
                    classes[predictions[j][i]] += self.alphas[j]
                except:
                    classes[predictions[j][i]] = self.alphas[j]
            finclass = 'a'
            m = 0
            for key in list(classes.keys()):
                if classes[key]>m:
                    finclass = key
                    m = classes[key]
            final_preds.append(finclass)
        return pd.Series(final_preds,dtype="category")
        

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        pass
