import numpy as np
import pandas as pd
from metrics import *
from tree.base import DecisionTree
from linearRegression import LinearRegression

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
        

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria,max_depth=1)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
# [fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

print('\nIRIS DATASET')

##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features
dataset = pd.read_csv("tree/iris.data",delimiter=",",header=None)
a = dataset[1]
b = dataset[3]
c = dataset[4]
d = list(zip(a,b,c))
np.random.shuffle(d)
frame = {}
frame['sepal width'] = []
frame['petal width'] = []
y = []
for i in range(len(d)):
    frame['sepal width'].append(d[i][0])
    frame['petal width'].append(d[i][1])
    y.append(d[i][2])
X = pd.DataFrame(frame)
y = pd.Series(y)
for i in range(len(y)):
    if (y[i]!='Iris-virginica'):
        y[i] = 'not virginica'
N = len(y)
t = int(np.floor(0.6*N))
X_train = X.iloc[:t,:]
y_train = y[:t]
X_test = X.iloc[t:,:]
y_test = list(y[t:])
y_test = pd.Series(y_test)
criteria = 'information_gain'
tree = DecisionTree(criterion=criteria,max_depth=1)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X_train, y_train)
y_hat = Classifier_AB.predict(X_test)
# [fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))

print("\nDECISION STUMP")
tree.fit(X_train,y_train,np.ones(N)/N)
y_hat = tree.predict(X_test)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))
