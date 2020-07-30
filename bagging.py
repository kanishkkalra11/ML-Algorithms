import numpy as np
import pandas as pd
import random

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        N = len(y)
        estimators = []
        for i in range(self.n_estimators):
            samples = [random.randint(0,N-1) for j in range(N)]
            vaai = []
            frame = {}
            for j in range(len(samples)):
                vaai.append(y[samples[j]])
                data = X.iloc[samples[j]]
                for item in list(X.columns):
                    try:
                        frame[item].append(data[item])
                    except:
                        frame[item] = []
                        frame[item].append(data[item])
            dataset = pd.DataFrame(frame)
            estimator = self.base_estimator
            estimator.fit(dataset,vaai)
            estimators.append(estimator)
        self.estimators = estimators
        return

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = []
        final_preds = []
        for i in range(len(self.estimators)):
            predictions.append(self.estimators[i].predict(X))
        for i in range(len(predictions[0])):
            classes = {}
            for j in range(len(predictions)):
                try:
                    classes[predictions[j][i]] += 1
                except:
                    classes[predictions[j][i]] = 1
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
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        pass
