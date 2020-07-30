import numpy as np
import pandas as pd
import random
from metrics import *
from tree.base import DecisionTree
# Or use sklearn decision tree
from sklearn import tree

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
N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
clf = tree.DecisionTreeClassifier()
Classifier_B = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators )
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
# [fig1, fig2] = Classifier_B.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



a = [1]*8 + [2]*8 + [3]*8 + [4]*8 + [5]*8 + [6]*8 + [7]*8 + [8]*8
b = [1,2,3,4,5,6,7,8]*8
c = [0,0,0,0,0,1,1,1]*2 + [0,0,1,0,0,1,1,1] + [0,0,0,0,0,1,1,1] + [0,0,0,0,0,1,1,0] + [1]*24

frame = {'x1':a,'x2':b}
X = pd.DataFrame(frame)
y = pd.Series(c)

clf = tree.DecisionTreeClassifier()
bag = BaggingClassifier(clf,5)
bag.fit(X,y)
y_hat = bag.predict(X)
print('\nAccuracy for dataset in slides: ', accuracy(y_hat, y))
