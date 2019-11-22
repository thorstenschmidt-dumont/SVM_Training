# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
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

"""
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

"""
"""
# The other test dataset
X_train, y_train = make_blobs(n_samples=1000, centers=2,
                        n_features=2, random_state=1)

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train, y_train)

x_train = x_train

y_train[y_train == 0] = -1
"""
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
x_train /= 25.5
x_test /= 25.5
x_test1 /= 25.5

# Redefining the labels
for i in range(len(y_test)):
    if y_test[i] == 8:
        y_test[i] = 1
    else:
        y_test[i] = 0
        
for i in range(len(y_test1)):
    if y_test1[i] == 8:
        y_test1[i] = 1
    else:
        y_test1[i] = 0

for i in range(len(y_train)):
    if y_train[i] == 8:
        y_train[i] = 1
    else:
        y_train[i] = 0
"""

# Importing the forest cover dataset
data = pd.read_csv("breast_cancer_wisconsin_clean.csv")
X = data.iloc[:, 1:9].values
y = data.iloc[:, 10].values

X = X.astype('float32')

# Feature scaling
#Max = 0
#for i in range(len(X[1])):
#    Max = X[:,i].max()
#    X[:, i] = X[:, i]/Max


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


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(C=100, kernel = 'rbf', degree = 4, random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""