#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:09:23 2019

@author: thorsten
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import math

class SMOModel:
    """Container object for the model used for sequential minimal optimization."""

    def __init__(self, X, y, C, kernel, alphas, b, errors):
        self.X = X               # training data vector
        self.y = y               # class label vector
        self.C = C               # regularization parameter
        self.kernel = kernel     # kernel function
        self.alphas = alphas     # lagrange multiplier vector
        self.b = b               # scalar bias term
        self.errors = errors     # error cache
        self._obj = []           # record of objective function value
        self.m = len(self.X)     # store size of training set

def linear_kernel(x, y, b=1):
    """Returns the linear combination of arrays `x` and `y` with
    the optional bias term `b` (set to 1 by default)."""

    return x @ y.T + b # Note the @ operator for matrix multiplication

def polynomial_kernel(x, y, b=1):
    """Calculate the Kernel value of x and y"""
    Result = ((x @ y.T)+b)**5

    return Result


def gaussian_kernel(x, y, sigma=1):
    """Returns the gaussian similarity of arrays `x` and `y` with
    kernel width parameter `sigma` (set to 1 by default)."""

    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    return result

def Kernel(x, y):
    """Calculate the Kernel value of x and y"""
    Result = (np.dot(x_train[x, :], x_train[y, :])+1)**3 # Polynomial
    # Result = (np.dot(x_train[x, :], x_train[y, :])+1) # Linear

    return Result

def KernelTest(x, y):
    """Calculate the Kernel value of x and y"""
    Result = (np.dot(x_test[x, :], x_train[y, :])+1)**3 # Polynomial
    #Result = (np.dot(x_test[x, :], x_train[y, :])+1) # Linear

    return Result

# Objective function to optimize
def objective_function(alphas, target, kernel, X_train):
    """Returns the SVM objective function based in the input model defined by:
    `alphas`: vector of Lagrange multipliers
    `target`: vector of class labels (-1 or 1) for training data
    `kernel`: kernel function
    `X_train`: training data for model."""

    return np.sum(alphas) - 0.5 * np.sum((target[:, None] * target[None, :]) * kernel(X_train, X_train) * (alphas[:, None] * alphas[None, :]))


# Decision function

def decision_function(alphas, target, kernel, X_train, x_test, b):
    """Applies the SVM decision function to the input feature vectors in `x_test`."""

    result = (alphas * target) @ kernel(X_train, x_test) - b
    return result

def plot_decision_boundary(model, ax, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
        """Plots the model's decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`)."""

        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate the model over the entire space
        xrange = np.linspace(model.X[:,0].min(), model.X[:,0].max(), resolution)
        yrange = np.linspace(model.X[:,1].min(), model.X[:,1].max(), resolution)
        grid = [[decision_function(model.alphas, model.y,
                                   model.kernel, model.X,
                                   np.array([xr, yr]), model.b) for xr in xrange] for yr in yrange]
        grid = np.array(grid).reshape(len(xrange), len(yrange))

        # Plot decision contours using grid and
        # make a scatter plot of training data
        ax.contour(xrange, yrange, grid, levels=levels, linewidths=(1, 1, 1),
                   linestyles=('--', '-', '--'), colors=colors)
        ax.scatter(model.X[:,0], model.X[:,1],
                   c=model.y, cmap=plt.cm.viridis, lw=0, alpha=0.25)

        # Plot support vectors (non-zero alphas)
        # as circled points (linewidth > 0)
        mask = np.round(model.alphas, decimals=2) != 0.0
        ax.scatter(model.X[mask,0], model.X[mask,1],
                   c=model.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')

        return grid, ax


def take_step(i1, i2, model):

    # Skip if chosen alphas are the same
    if i1 == i2:
        return 0, model

    alph1 = model.alphas[i1]
    alph2 = model.alphas[i2]
    y1 = model.y[i1]
    y2 = model.y[i2]
    E1 = model.errors[i1]
    E2 = model.errors[i2]
    s = y1 * y2

    # Compute L & H, the bounds on new possible alpha values
    if (y1 != y2):
        L = max(0, alph2 - alph1)
        H = min(model.C, model.C + alph2 - alph1)
    elif (y1 == y2):
        L = max(0, alph1 + alph2 - model.C)
        H = min(model.C, alph1 + alph2)
    if (L == H):
        return 0, model

    # Compute kernel & 2nd derivative eta
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])
    eta = 2 * k12 - k11 - k22

    # Compute new alpha 2 (a2) if eta is negative
    if (eta < 0):
        a2 = alph2 - y2 * (E1 - E2) / eta
        # Clip a2 based on bounds L & H
        if L < a2 < H:
            a2 = a2
        elif (a2 <= L):
            a2 = L
        elif (a2 >= H):
            a2 = H

    # If eta is non-negative, move new a2 to bound with greater objective function value
    else:
        alphas_adj = model.alphas.copy()
        alphas_adj[i2] = L
        # objective function output with a2 = L
        Lobj = objective_function(alphas_adj, model.y, model.kernel, model.X)
        alphas_adj[i2] = H
        # objective function output with a2 = H
        Hobj = objective_function(alphas_adj, model.y, model.kernel, model.X)
        if Lobj > (Hobj + eps):
            a2 = L
        elif Lobj < (Hobj - eps):
            a2 = H
        else:
            a2 = alph2

    # Push a2 to 0 or C if very close
    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (model.C - 1e-8):
        a2 = model.C

    # If examples can't be optimized within epsilon (eps), skip this pair
    if (np.abs(a2 - alph2) < eps * (a2 + alph2 + eps)):
        return 0, model

    # Calculate new alpha 1 (a1)
    a1 = alph1 + s * (alph2 - a2)

    # Update threshold b to reflect newly calculated alphas
    # Calculate both possible thresholds
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b

    # Set new threshold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 and a1 < C:
        b_new = b1
    elif 0 < a2 and a2 < C:
        b_new = b2
    # Average thresholds if both are bound
    else:
        b_new = (b1 + b2) * 0.5

    # Update model object with new alphas & threshold
    model.alphas[i1] = a1
    model.alphas[i2] = a2

    # Update error cache
    # Error cache for optimized alphas is set to 0 if they're unbound
    for index, alph in zip([i1, i2], [a1, a2]):
        if 0.0 < alph < model.C:
            model.errors[index] = 0.0

    # Set non-optimized errors based on equation 12.11 in Platt's book
    non_opt = [n for n in range(model.m) if (n != i1 and n != i2)]
    model.errors[non_opt] = model.errors[non_opt] + \
                            y1*(a1 - alph1)*model.kernel(model.X[i1], model.X[non_opt]) + \
                            y2*(a2 - alph2)*model.kernel(model.X[i2], model.X[non_opt]) + model.b - b_new

    # Update model threshold
    model.b = b_new

    return 1, model


def examine_example(i2, model):

    y2 = model.y[i2]
    alph2 = model.alphas[i2]
    E2 = model.errors[i2]
    r2 = E2 * y2

    # Proceed if error is within specified tolerance (tol)
    if ((r2 < -tol and alph2 < model.C) or (r2 > tol and alph2 > 0)):

        if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
            # Use 2nd choice heuristic is choose max difference in error
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

        # Loop through non-zero and non-C alphas, starting at a random point
        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                          np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

        # loop through all alphas, starting at a random point
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

    return 0, model


def train(model):

    numChanged = 0
    examineAll = 1

    while(numChanged > 0) or (examineAll):
        numChanged = 0
        if examineAll:
            # loop over all training examples
            for i in range(model.alphas.shape[0]):
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
                if examine_result:
                    obj_result = objective_function(model.alphas, model.y, model.kernel, model.X)
                    model._obj.append(obj_result)
        else:
            # loop over examples where alphas are not already at their limits
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
                if examine_result:
                    obj_result = objective_function(model.alphas, model.y, model.kernel, model.X)
                    model._obj.append(obj_result)
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1

    return model

"""
X_train, y_train = make_blobs(n_samples=1000, centers=2,
                        n_features=2, random_state=1)

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train, y_train)

y_train[y_train == 0] = -1
"""
"""
# Self generated sin wave dataset
x_train = np.random.rand(1000,2)*2-1
x_train[:,0] = x_train[:,0]*2*math.pi
x_train[:,1] = x_train[:,1]*1.1
y_train = np.zeros(1000)
for i in range(len(y_train)):
    if x_train[i,1] > math.sin(x_train[i,0]):
        y_train[i] = 1
    else:
        y_train[i] = -1

x_test = np.random.rand(1000,2)*2-1
x_test[:,0] = x_test[:,0]*2*math.pi
x_test[:,1] = x_test[:,1]*1.1
y_test = np.zeros(1000)
for i in range(len(x_test)):
    if x_test[i,1] > math.sin(x_test[i,0]):
        y_test[i] = 1
    else:
        y_test[i] = -1

#scaler = StandardScaler()
#x_train = scaler.fit_transform(X_train, y_train)

#x_train = X_train
"""
"""
data = pd.read_csv("sonar-all-data.csv")

x_train = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values
for i in range(len(y)):
    if y[i] == "R":
        y[i] = 1
    else:
        y[i] = -1

y_train = y.astype(int)
"""
"""
# Importing the forest cover dataset
data =pd.read_csv("covtype.csv")
X = data.iloc[:, 0:53].values
y = data.iloc[:, 54].values

X = X.astype('float32')

Max = 0
for i in range(len(X[1])):
    Max = X[:,i].max()
    X[:, i] = X[:, i]/(Max*100)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#X_test = sc.transform(X_test)

y_train[y_train != 1] = -1
"""
"""
# Importing the dataset
(X_train, Y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test1, y_train, y_test1 = train_test_split(X_train, Y_train, test_size = 50/60, random_state = 0)

# Data preparation and matrix reshaping
x_train = x_train.transpose(2, 0, 1).reshape(-1, x_train.shape[0])
x_train = np.transpose(x_train)

x_test = x_test.transpose(2, 0, 1).reshape(-1, x_test.shape[0])
x_test = np.transpose(x_test)

# Redefine matrix as float for division operation
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# Normalising to between 0 and 0.01
x_train /= 255
x_test /= 255

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Redefining the labels
for i in range(len(y_test)):
    if y_test[i] == 8:
        y_test[i] = 1
    else:
        y_test[i] = -1

for i in range(len(y_train)):
    if y_train[i] == 8:
        y_train[i] = 1
    else:
        y_train[i] = -1

# Create list of all nonzero entries
List = [[] for i in range(len(x_train))]
for i in range(len(x_train)):
    for j in range(784):
        if x_train[i, j] != 0:
            List[i].append(j)
"""
dataset_train = pd.read_csv('mnist_train.csv')
dataset_test = pd.read_csv('mnist_test.csv')
x_train = dataset_train.iloc[0:9999, 1:].to_numpy()
y_train = dataset_train.iloc[0:9999, 0].to_numpy()

x_test = dataset_test.iloc[:, 1:].to_numpy()
y_test = dataset_test.iloc[:, 0].to_numpy()

# Normalising to between 0 and 0.1
x_train = x_train/2550
x_test = x_test/2550
# Redefining the labels
for i in range(len(y_test)):
    if y_test[i] == 8:
        y_test[i] = 1
    else:
        y_test[i] = -1

for i in range(len(y_train)):
    if y_train[i] == 8:
        y_train[i] = 1
    else:
        y_train[i] = -1

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int')
y_test = y_test.astype('int')

"""
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
X_test = sc.transform(X_test)

y_train[y_train == 0] = -1
"""
"""
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
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)#random_state = 0)

y_train[y_train == 4] = 1
y_train[y_train == 2] = -1

y_test[y_test == 4] = 1
y_test[y_test == 2] = -1
"""
# Set model parameters and initial values
C = 100
m = len(x_train)
initial_alphas = np.zeros(m)
initial_b = 0.0

# Set tolerances
tol = 0.001 # error tolerance
eps = 0.1 # alpha tolerance

# Instantiate model
model = SMOModel(x_train, y_train, C, polynomial_kernel,
                 initial_alphas, initial_b, np.zeros(m))

# Initialize error cache
initial_error = decision_function(model.alphas, model.y, model.kernel,
                                  model.X, model.X, model.b) - model.y
model.errors = initial_error

np.random.seed(0)
output = train(model)


#fig, ax = plt.subplots()
#grid, ax = plot_decision_boundary(output, ax)

output.alphas.sum()
SVs = output.alphas.nonzero()

SVs = SVs[0]

result = decision_function(output.alphas, output.y, output.kernel,
                                  output.X, output.X, output.b) - model.y


PositiveT = 0
NegativeT = 0
PositiveF = 0
NegativeF = 0
for i in range(len(x_train)):
    if result[i] > 0 and y_train[i] == 1:
        PositiveT = PositiveT + 1
    elif result[i] > 0 and y_train[i] == -1:
        PositiveF = PositiveF + 1
    elif result[i] < 0 and y_train[i] == -1:
        NegativeT = NegativeT + 1
    else:
        NegativeF = NegativeF + 1

print(PositiveT, PositiveF, NegativeT, NegativeF)

#for i in range(len(SVs)):
#    print(SVs[i], output.alphas[int(SVs[i])])

check1 = (model.alphas * y_train)
check2 = polynomial_kernel(x_train, x_test)
result = check1 @ check2 - model.b - y_test

PositiveT = 0
NegativeT = 0
PositiveF = 0
NegativeF = 0
for i in range(len(x_test)):
    if result[i] > 0 and y_test[i] == 1:
        PositiveT = PositiveT + 1
    elif result[i] > 0 and y_test[i] == -1:
        PositiveF = PositiveF + 1
    elif result[i] < 0 and y_test[i] == -1:
        NegativeT = NegativeT + 1
    else:
        NegativeF = NegativeF + 1

print(PositiveT, PositiveF, NegativeT, NegativeF)

"""
check1 = (model.alphas * y_train)
check2 = polynomial_kernel(x_train, x_test)
result = check1 @ check2 - model.b  - y_test
PositiveT = 0
NegativeT = 0
PositiveF = 0
NegativeF = 0
for i in range(len(x_test)):
    if result[i] > 0 and y_test[i] == 1:
        PositiveT = PositiveT + 1
    elif result[i] > 0 and y_test[i] == -1:
        PositiveF = PositiveF + 1
    elif result[i] < 0 and y_test[i] == -1:
        NegativeT = NegativeT + 1
    else:
        NegativeF = NegativeF + 1
"""