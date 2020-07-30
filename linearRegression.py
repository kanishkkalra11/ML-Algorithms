import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
import autograd 
import autograd.numpy as npy  # Thinly-wrapped numpy
from autograd import grad 
import math

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.thetas = None

    def split(self,dfm, chunk_size):
        indices = self.index_marks(dfm.shape[0], chunk_size)
        return np.split(dfm, indices)

    def index_marks(self,nrows, chunk_size):
        return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)
    
    def mse(self, X, y, theta):
        error = 0
        y_pred = npy.dot(theta, X.T)  
        for i in range(len(y)):
            error= error + (y[i]-y_pred[i])**2
            mse = error / len(y)
        return mse

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        if (self.fit_intercept):
            a = pd.Series(np.ones(len(y)))
            X = pd.concat([a, X], axis=1,  ignore_index=True).reset_index(drop=True)
          
        num_features = len(X.columns) 
        dataset = pd.concat([X,y], axis=1,  ignore_index=True).reset_index(drop=True)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        #create batches
        batches = self.split(dataset, batch_size)    
        theta = np.zeros(num_features)
        iters = np.zeros((n_iter, num_features))
        for i in range(1,n_iter+1):        
            for j in range(len(batches)):
                X = batches[j].iloc[:, 0:num_features]
                y = batches[j].iloc[:, num_features]
                y_pred = np.dot(theta, X.T)  
                for k in range(len(theta)):
                    theta_k = (-2/len(batches[j])) * sum(X.iloc[:,k] * (y- y_pred))  
                    if lr_type == 'constant':
                        theta[k] = theta[k] - lr * theta_k  
                    else:
                        theta[k] = theta[k] - (lr/i)* theta_k  
            iters[i-1] = theta
        self.coef_ = theta
        self.thetas = iters
        return

    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        if (self.fit_intercept):
            a = pd.Series(np.ones(len(y)))
            X = pd.concat([a, X], axis=1,  ignore_index=True).reset_index(drop=True)
        num_features = len(X.columns) 
        dataset = pd.concat([X,y], axis=1,  ignore_index=True).reset_index(drop=True)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        #create batches
        batches = self.split(dataset, batch_size) 
        theta = np.zeros(len(X.columns))
        iters = np.zeros((n_iter, num_features))
        for i in range(1,n_iter+1): 
            for j in range(len(batches)):
                X = batches[j].iloc[:, 0:num_features]
                y = batches[j].iloc[:, num_features]
                y_pred = np.dot(theta, X.T)

                X = X.mul(y- y_pred,  axis = 0) 
                theta_j =(-2/len(batches[j])) * X.sum(axis = 0, skipna = True) 
                if lr_type == 'constant':
                    theta = np.subtract(theta, lr*theta_j)
                else:
                    theta = np.subtract(theta , (lr/i)*theta_j) 
            iters[i-1] = theta
        self.coef_ = theta
        self.thetas = iters
        return

    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        if(self.fit_intercept):
            a = pd.Series(np.ones(len(y)))
            X = pd.concat([a, X], axis=1,  ignore_index=True).reset_index(drop=True)
     
        num_features = len(X.columns) 
        dataset = pd.concat([X,y], axis=1,  ignore_index=True).reset_index(drop=True)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        #create batches
        batches = self.split(dataset, batch_size) 
        theta = np.zeros(len(X.columns))
        iters = np.zeros((n_iter, num_features))
        for i in range(1,n_iter+1): 
            for j in range(len(batches)):
                X = batches[j].iloc[:, 0:num_features]
                y = batches[j].iloc[:, num_features] 
                for k in range(len(theta)):
                    theta_k = grad(self.mse, 2)
                    if lr_type == 'constant':
                        theta[k] = theta[k] - lr *(theta_k(np.array(X),np.array(y), theta)[k])
                    else:
                        theta[k] = theta[k] - (lr/i)* (theta_k(np.array(X),np.array(y), theta)[k])
            iters[i-1] = theta
        self.coef_ = theta
        self.thetas = iters
        return

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''

        if(self.fit_intercept):
            a = pd.Series(np.ones(len(y)))
            X = pd.concat([a, X], axis=1,  ignore_index=True).reset_index(drop=True)
        X = X.values
        X_T = np.transpose(X)
        try:
            theta = np.matmul(np.linalg.inv(np.matmul(X_T,X)),np.matmul(X_T,y))
        except:
            theta = np.matmul(np.linalg.pinv(np.matmul(X_T,X)),np.matmul(X_T,y))
        self.coef_ = theta
        return

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''

        if(self.fit_intercept):
            a = pd.Series(np.ones(X.shape[0]))
            X = pd.concat([a, X], axis=1,  ignore_index=True).reset_index(drop=True)
        X = X.values
        y = np.matmul(X,self.coef_)
        return pd.Series(y)

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """

        max_coef=6

        if(self.fit_intercept):
            a = pd.Series(np.ones(len(y)))
            X = pd.concat([a, X], axis=1,  ignore_index=True).reset_index(drop=True)

        theta0 = np.linspace(-max_coef, max_coef, 100)
        theta1 = np.linspace(-max_coef, max_coef, 100)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0, theta1)
        errors = []
        for i,j in zip(theta0_mesh,theta1_mesh):
                errors.append(np.sum(np.square(y.as_matrix().reshape(len(y),1) - np.dot(X,pd.DataFrame([i,j]))),axis=0))
        errors = np.array(errors)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        err_point = np.sum(np.square(y.as_matrix().reshape(len(y),1) - np.dot(X,pd.DataFrame([[t_0], [t_1]]))))
        ax.plot_surface(theta0_mesh, theta1_mesh, errors,cmap='viridis', edgecolor='none', alpha=0.725)
        ax.scatter3D(t_0, t_1, err_point, c='r',s=np.pi*4)
        return 

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        x_max = np.max(X) + 1
        x_min = np.min(X) - 1
        #calculating line values of x and y
        x = np.linspace(x_min, x_max, 1000)
        y_dash = t_0 + t_1*x
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(x, y_dash, color='#00ff00', label='Linear Regression')
        ax.scatter(X.iloc[:,0], y, color='#ff0000', label='Data Point')
        return

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        max_coef=20
        if(self.fit_intercept):
            a = pd.Series(np.ones(len(y)))
            X = pd.concat([a, X], axis=1,  ignore_index=True).reset_index(drop=True)
        theta0 = np.linspace(-max_coef, max_coef, 100)
        theta1 = np.linspace(-max_coef, max_coef, 100)
        theta0_mesh, theta_1_mesh = np.meshgrid(theta0, theta1)
        errors = []
        for i,j in zip(theta0_mesh,theta1_mesh):
                errors.append(np.sum(np.square(y.as_matrix().reshape(len(y),1) - np.dot(X,pd.DataFrame([i,j]))),axis=0))
        errors = np.array(errors)
        fig = plt.figure()
        ax = plt.axes()
        err_point = np.sum(np.square(y.as_matrix().reshape(len(y),1) - np.dot(X,pd.DataFrame([[t_0], [t_1]]))))
        ax.contour(theta0_mesh, theta1_mesh, errors, 20,cmap='viridis', edgecolor='none')
        ax.scatter(t_0, t_1,c='r',s=np.pi*4)
        return
