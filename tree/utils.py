import numpy as np
import pandas as pd


def entropy(Y,weights):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    
    y = list(Y)
    w = list(weights)
    classes = {}
    for i in range(len(y)):
        try:
            classes[y[i]] += w[i]
        except:
            classes[y[i]] = w[i]
    entropy = 0
    for key in classes.keys():
        p = classes[key]
        entropy += -p * np.log2(p)
    
    return entropy

def gini_index(Y,weights):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    y = list(Y)
    w = list(weights)
    classes = {}
    for i in range(len(y)):
        try:
            classes[y[i]] += w[i]
        except:
            classes[y[i]] = w[i]
    temp = 0
    for key in classes.keys():
        p = classes[key]
        temp += p * p
    return (1-temp)

def information_gain(Y, attr, weights):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    ent = entropy(pd.Series(Y),weights)
    total = len(Y)
    Y = list(Y)
    attrtemp = list(attr)
    w = list(weights)
    attr, Y = (list(t) for t in zip(*sorted(zip(attrtemp, Y))))
    _, w = (list(t) for t in zip(*sorted(zip(attrtemp, w))))
    inf_gain = -np.inf
    for i in range(len(attr)-1):
        temp = (attr[i] + attr[i+1])/2
        ent1 = np.sum(w[:i+1]) * entropy(pd.Series(Y[:i+1]),w[:i+1])
        ent2 = np.sum(w[i+1:]) * entropy(pd.Series(Y[i+1:]),w[i+1:])
        gain = ent - ent1 - ent2
        if (gain>inf_gain):
            inf_gain = gain
            split_point = temp
            split_index = i
    return inf_gain , split_point, split_index

def gini_gain(Y,attr,weights):
    total = len(Y)
    Y = list(Y)
    attrtemp = list(attr)
    w = list(weights)
    attr, Y = (list(t) for t in zip(*sorted(zip(attrtemp, Y))))
    _, w = (list(t) for t in zip(*sorted(zip(attrtemp, w))))
    gin_gain = np.inf
    for i in range(len(attr)-1):
        temp = (attr[i] + attr[i+1])/2
        gain1 = np.sum(w[:i+1]) * gini_index(pd.Series(Y[:i+1]),w[:i+1])
        gain2 = np.sum(w[i+1:]) * gini_index(pd.Series(Y[i+1:]),w[i+1:])
        gain = gain1 + gain2
        if (gain<gin_gain):
            gin_gain = gain
            split_point = temp
            split_index = i
    return gin_gain , split_point, split_index