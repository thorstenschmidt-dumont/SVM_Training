#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:35:27 2019

@author: thorsten
"""

import numpy as np
import random
import functools
import math
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import time
import idx2numpy
import tensorflow as tf
from operator import itemgetter
import numdifftools as nd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import pandas as pd

def Kernel(x, y):
    """Calculate the Kernel value of x and y"""

    #Result = (np.dot(x_test[x, :], x_train[y, :])+1)**5 # Polynomial
    # Result = (np.dot(x_train[x, :], x_train[y, :])+1) # Linear
    # Sum = DotProduct(x, y)
    #Sum = 0.0
    #for i in range(2):
    #    Sum = Sum + x_train[x, i]*x_train[y, i]
    # Result = (Sum+1)**5

    #Gaussian
    sigma = 1
    if np.ndim(x_test[x, :]) == 1 and np.ndim(x_train[y, :]) == 1:
        Result = np.exp(- (np.linalg.norm(x_test[x, :] - x_train[y, :], 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x_test[x, :]) > 1 and np.ndim(x_train[y, :]) == 1) or (np.ndim(x_test[x, :]) == 1 and np.ndim(x_train[y, :]) > 1):
        Result = np.exp(- (np.linalg.norm(x_test[x, :] - x_train[y, :], 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x_test[x, :]) > 1 and np.ndim(x_train[y, :]) > 1:
        Result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))

    return Result
"""
# Importing the dataset
(x_train, y_train), (x_test1, y_test1) = tf.keras.datasets.mnist.load_data()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 50/60, random_state = 0)

# Data preparation and matrix reshaping
x_train = x_train.transpose(2, 0, 1).reshape(-1, x_train.shape[0])
x_train = np.transpose(x_train)

x_test = x_test.transpose(2, 0, 1).reshape(-1, x_test.shape[0])
x_test = np.transpose(x_test)

x_test1 = x_test1.transpose(2, 0, 1).reshape(-1, x_test1.shape[0])
x_test1 = np.transpose(x_test1)

# Redefine matrix as float for division operation
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_test1 = x_test1.astype('float32')
y_train = y_train.astype('int')
y_test = y_test.astype('int')
y_test1 = y_test1.astype('int')

# Normalising to between 0 and 0.1
x_train /= 2550
x_test /= 2550
x_test1 /= 2550

# Redefining the labels
for i in range(len(y_test)):
    if y_test[i] == 8:
        y_test[i] = 1
    else:
        y_test[i] = -1

for i in range(len(y_test1)):
    if y_test1[i] == 8:
        y_test1[i] = 1
    else:
        y_test1[i] = -1

for i in range(len(y_train)):
    if y_train[i] == 8:
        y_train[i] = 1
    else:
        y_train[i] = -1
"""
#x_test = X_test

# Importing the breast cancer dataset
data = pd.read_csv("breast_cancer_wisconsin_clean.csv")
X = data.iloc[:, 1:9].values
y = data.iloc[:, 10].values

X = X.astype('float32')

# Feature scaling
Max = 0
for i in range(len(X[1])):
    Max = X[:,i].max()
    X[:, i] = X[:, i]/(3.25*Max)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#X_test = sc.transform(X_test)

y_train[y_train == 4] = 1
y_train[y_train == 2] = -1

y_test[y_test == 4] = 1
y_test[y_test == 2] = -1

PositiveT = 0
NegativeT = 0
PositiveF = 0
NegativeF = 0
for j in range(len(x_test)):
    Sum = 0.0
    for i in range(len(SVs)):
        Sum = Sum + y_train[int(SVs[i])]*alpha[int(SVs[i])]*Kernel(j, int(SVs[i]))
    Classification = Sum + b
    if Classification > 0 and y_test[j] == 1:
        PositiveT = PositiveT + 1
    elif Classification > 0 and y_test[j] == -1:
        PositiveF = PositiveF + 1
    elif Classification < 0 and y_test[j] == -1:
        NegativeT = NegativeT + 1
    else:
        NegativeF = NegativeF + 1

"""
PositiveT = 0
NegativeT = 0
PositiveF = 0
NegativeF = 0
for j in range(len(x_train)):
    Sum = 0.0
    for i in range(len(SVs)):
        Sum = Sum + y_train[int(SVs[i])]*alpha[int(SVs[i])]*Kernel(j, int(SVs[i]))
    Classification = Sum + b
    if Classification > 0 and y_train[j] == 1:
        PositiveT = PositiveT + 1
    elif Classification > 0 and y_train[j] == -1:
        PositiveF = PositiveF + 1
    elif Classification < 0 and y_train[j] == -1:
        NegativeT = NegativeT + 1
    else:
        NegativeF = NegativeF + 1
"""