import numpy as np
import pandas as pd

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    corr = 0
    for i in range(len(y)):
        if(y[i]==y_hat[i]):
            corr += 1
    return corr/len(y)

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    precise = 0
    tot = 0
    for i in range(len(y)):
        if (y_hat[i]==cls):
            tot += 1
            if (y[i]==cls):
                precise += 1
    if(tot==0):
        return 0
    return precise/tot


def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    recalled = 0
    tot = 0
    for i in range(len(y)):
        if (y[i]==cls):
            tot += 1
            if (y_hat[i]==cls):
                recalled += 1
    return recalled/tot



def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    tot = 0
    for i in range(len(y)):
        e = y_hat[i] - y[i]
        s = e**2
        tot += s
    return np.sqrt(tot/len(y))


def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    tot = 0
    for i in range(len(y)):
        e = y[i] - y_hat[i]
        a = np.abs(e)
        tot += a
    return tot/len(y)