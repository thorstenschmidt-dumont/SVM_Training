#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:18:06 2019

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

"""
def Kernel(x, y):
    Calculate the Kernel value of x and y
    # Result = (np.dot(x, y)+1)**5
    Result = (np.dot(x, y)+1)**5
    return Result
"""


def DotProduct(x, y):
    
    n1 = len(List[x])
    n2 = len(List[y])
    p1 = 0
    p2 = 0
    dot = 0.0
    x1 = np.array(List[x])
    y1 = np.array(List[y])
    while p1 < n1 and p2 < n2:
        a1 = x1[p1]
        a2 = y1[p2]
        if a1 == a2:
            dot = dot + x_train[x, a1]*x_train[y, a2]
            p1 += 1
            p2 += 1
        elif a1 > a2:
            p2 += 1
        else:
            p1 += 1
    
    # dot = np.dot(x_train[x, :], x_train[y, :])
    return dot
    
"""
def Kernel(x, y):
    Calculate the Kernel value of x and y
    # Result = (np.dot(x, y)+1)**5
    
    Sum = 0.0
    for j in range(len(x)):
        Sum = Sum + x[j]*y[j]
    Result = (Sum+1)**5
    
    return Result
"""


def Kernel(x, y):
    """Calculate the Kernel value of x and y"""

    Result = (np.dot(x_train[x, :], x_train[y, :])+1)**5 # Polynomial
    #Result = (np.dot(x_train[x, :], x_train[y, :])+1) # Linear
    #Gaussian
    """
    sigma = 1
    if np.ndim(x_train[x, :]) == 1 and np.ndim(x_train[y, :]) == 1:
        Result = np.exp(- (np.linalg.norm(x_train[x, :] - x_train[y, :], 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x_train[x, :]) > 1 and np.ndim(x_train[y, :]) == 1) or (np.ndim(x_train[x, :]) == 1 and np.ndim(x_train[y, :]) > 1):
        Result = np.exp(- (np.linalg.norm(x_train[x, :] - x_train[y, :], 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x_train[x, :]) > 1 and np.ndim(x_train[y, :]) > 1:
        Result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    """
    return Result


def KernelTest(x, y):
    """Calculate the Kernel value of x and y"""

    Result = (np.dot(x_test[x, :], x_train[y, :])+1)**5 # Polynomial
    # Result = (np.dot(x_train[x, :], x_train[y, :])+1) # Linear
    # Sum = DotProduct(x, y)
    #Sum = 0.0
    #for i in range(2):
    #    Sum = Sum + x_train[x, i]*x_train[y, i]
    # Result = (Sum+1)**5
    """
    #Gaussian
    sigma = 1
    if np.ndim(x_test[x, :]) == 1 and np.ndim(x_train[y, :]) == 1:
        Result = np.exp(- (np.linalg.norm(x_test[x, :] - x_train[y, :], 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x_test[x, :]) > 1 and np.ndim(x_train[y, :]) == 1) or (np.ndim(x_test[x, :]) == 1 and np.ndim(x_train[y, :]) > 1):
        Result = np.exp(- (np.linalg.norm(x_test[x, :] - x_train[y, :], 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x_test[x, :]) > 1 and np.ndim(x_train[y, :]) > 1:
        Result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    """
    return Result


def countX(lst, x):
    """Count number of occurences of x"""
    return lst.count(x)


def GenerateInitialSolution():
    """Generate an initial feasible solution"""
    c = random.random()*C
    count = 0
    while np.count_nonzero(alpha) < gamma:
        rand = random.randint(0, len(x_train)-1)
        if y_train[rand] == 1:
            alpha[rand] = c
            L[rand, 1] = c
            # L[count, 0] = rand
            # L[count, 1] = alpha[rand]
            SVs[count] = rand
            count += 1
    while np.count_nonzero(alpha) < 2*gamma:
        rand = random.randint(0, len(x_train)-1)
        if y_train[rand] == -1:
            alpha[rand] = c
            L[rand, 1] = c
            # L[count, 0] = rand
            # L[count, 1] = alpha[rand]
            SVs[count] = rand
            count += 1
    return alpha

def GenerateS(s):
    """Compute the initial value of S"""    
    for i in range(len(x_train)):
        for j in range(len(SVs)):
            index = int(SVs[j])
            s[i] = s[i] + alpha[index]*y_train[index] * Kernel(i, index) # (DotProduct(i, index)+1)**5
    return s


def UpdateS(s, Difference, WorkingSet):
    """Compute the updated value of S"""
    for i in range(len(x_train)):
        Sum = 0.0
        for j in range(q):
            Sum = Sum + (Difference[j])*y_train[int(WorkingSet[j,0])]*Kernel(i, int(WorkingSet[j,0]))
        s[i] = s[i] + Sum
    return s

def UpdateS1SVs(s, Difference, WorkingSet):
    """Compute the updated S for support vectors"""
    


def UpdateS1(i):
    """Compute the updated value of S"""
    Sum = 0.0
    for j in range(q):
        Sum1 = Kernel(i, int(WorkingSet[j,0]))
        Sum = Sum + (Difference[j])*y_train[int(WorkingSet[j,0])]*Sum1
    s1[i] = s1[i] + Sum
    return s1[i]


def CalculateG(g, s):
    """Compute the values of g"""
    for i in range(len(y_train)):
        g[i] = 1 - y_train[i]*s[i]
    return g


def SortL(L, g):
    """Sorts the list of support vectors in ascending order according to yigi"""
    for i in range(len(L)):
        L[i, 0] = i
        L[i, 1] = alpha[i]
        L[i, 2] = y_train[i]*g[i]
    L = sorted(L, key=itemgetter(2))
    L = np.array(L)
    return L


"""
def SelectWorkingSet(L, q):
    Selects the working set from L such that the specific condistions hold
    for i in range(int(q/2)):
        # print(len(L))
        WorkingSet[i, 0] = L[i, 0]
        if L[i, 1] > 0 and L[i, 1] < C:
            WorkingSet[i, 1] = L[i, 1]
        elif y_train[int(L[i, 0])] == -1:
            WorkingSet[i, 1] = 0
        elif y_train[int(L[i, 0])] == 1:
            WorkingSet[i, 1] = C

    for i in range(int(q/2)):
        index = int(len(L) - i - 1)
        WorkingSet[int(i+q/2), 0] = L[index, 0]
        if L[index, 1] > 0 + 0.001 and L[index, 1] < C:
            WorkingSet[int(i+q/2), 1] = L[index, 1]
        elif y_train[int(L[index, 0])] == -1:
            WorkingSet[i+int(q/2), 1] = C
        elif y_train[int(L[index, 0])] == 1:
            WorkingSet[i+int(q/2), 1] = 0
    # print(WorkingSet)
    return WorkingSet
"""


def SelectWorkingSet(L, q):
    """Selects the working set from L such that the specific condistions hold"""
    i = 0
    index = 0
    while i < int(q/2):
        if L[index, 1] > 0 and L[index, 1] < C:
            WorkingSet[i, 0] = L[index, 0]
            WorkingSet[i, 1] = L[index, 1]
            WorkingSet[i, 2] = index
            i = i + 1
        elif y_train[int(L[index, 0])] == -1 and L[index, 1] <= 0:
            WorkingSet[i, 0] = L[index, 0]
            WorkingSet[i, 1] = L[index, 1]
            WorkingSet[i, 2] = index
            i = i + 1
        elif y_train[int(L[index, 0])] == 1 and L[index, 1] == 100:
            WorkingSet[i, 0] = L[index, 0]
            WorkingSet[i, 1] = L[index, 1]
            WorkingSet[i, 2] = index
            i = i + 1
        index = index + 1
        # print(WorkingSet)
        # print(index)

    index = len(y_train) - 1
    while i < int(q):
        j = 0
        while j < (int(q/2)):
            if index == int(WorkingSet[j, 2]):
                # print("Hello cunt")
                # print(index)
                index = index - 1
                # print(index)
                if j > 0:
                    j = 0
            else:
                j = j + 1
        if L[index, 1] > 0 + error and L[index, 1] < C:
            WorkingSet[i, 0] = L[index, 0]
            WorkingSet[i, 1] = L[index, 1]
            WorkingSet[i, 2] = index
            i = i + 1
        elif y_train[int(L[index, 0])] == 1 and L[index, 1] <= 0:
            WorkingSet[i, 0] = L[index, 0]
            WorkingSet[i, 1] = L[index, 1]
            WorkingSet[i, 2] = index
            i = i + 1
        elif y_train[int(L[index, 0])] == -1 and L[index, 1] == 100:
            WorkingSet[i, 0] = L[index, 0]
            WorkingSet[i, 1] = L[index, 1]
            WorkingSet[i, 2] = index
            i = i + 1
        index = index - 1
        # print(WorkingSet)
    return WorkingSet


def UpdateL(L, WorkingSet):
    for i in range(q): #range(int(q/2)):
        index = int(WorkingSet[i, 2])
        L[index, 1] = WorkingSet[i, 1]
        # if WorkingSet[i, 1] == 0:
        #    L = np.delete(L, i, 0)
    # for i in range(int(q/2)):
        # index = int(len(L) - i - 1)
        # L[index, 1] = WorkingSet[int(i+q/2), 1]
        # if WorkingSet[int(i+q/2), 1] == 0:
        #    L = np.delete(L, index, 0)
    return L


def CalculateqBN(qBN, WorkingSet):
    for i in range(q):
        Sum = 0.0
        for j in range(q):
            Sum1 = Kernel(int(WorkingSet[i,0]), int(WorkingSet[j,0])) # (DotProduct(int(WorkingSet[i,0]), int(WorkingSet[j,0]))+1)**5
            Sum = Sum + alpha[int(WorkingSet[j, 0])]*y_train[int(WorkingSet[j, 0])]*Sum1
        qBN[i] = y_train[int(WorkingSet[i, 0])]*(s[int(WorkingSet[i, 0])]-Sum)
    return qBN


def CalculateQBB(QBB, WorkingSet):
    """
    for i in range(q):
        for j in range(q):
            QBB[i, j] = y_train[int(WorkingSet[i, 0])]*y_train[int(WorkingSet[j, 0])]*Kernel(x_train[int(WorkingSet[i, 0]), :], x_train[int(WorkingSet[j, 0]), :])
    """
    """
    for i in range(q):
        QBB[i,i] = 20*(np.dot(x_train[int(WorkingSet[i, 0]), :],x_train[int(WorkingSet[i, 0]), :])+1)**3*sum(x_train[int(WorkingSet[i, 0]), :])**2
    # print(QBB)
    for i in range(q):
        j = i+1
        while j < q:
            QBB[i,j] = 20*y_train[int(WorkingSet[i, 0])]*y_train[int(WorkingSet[j, 0])]*(np.dot(x_train[int(WorkingSet[i, 0]), :],x_train[int(WorkingSet[j, 0]), :])+1)**3*sum(x_train[int(WorkingSet[i, 0]), :])*sum(x_train[int(WorkingSet[j, 0]), :]) + 5*y_train[int(WorkingSet[i, 0])]*y_train[int(WorkingSet[j, 0])]*(np.dot(x_train[int(WorkingSet[i, 0]), :],x_train[int(WorkingSet[j, 0]), :])+1)**4
            #QBB[i, j] = 5*y_train[int(WorkingSet[i, 0])]*y_train[int(WorkingSet[j, 0])]*(np.dot(x_train[int(WorkingSet[i, 0]), :],x_train[int(WorkingSet[j, 0]), :])+1)**4
            #QBB[i, j] = (np.dot(x_train[int(WorkingSet[i, 0]), :],x_train[int(WorkingSet[j, 0]), :])+1)**4*sum(x_train[int(WorkingSet[i, 0]), :])*sum(x_train[int(WorkingSet[j, 0]), :])
            j += 1
    """
    
    for i in range(q):
        QBB[i, i] = Kernel(int(WorkingSet[i, 0]),int(WorkingSet[i, 0]))# (DotProduct(int(WorkingSet[i, 0]),int(WorkingSet[i, 0]))+1)**5 # (np.dot(x_train[int(WorkingSet[i, 0]), :],x_train[int(WorkingSet[i, 0]), :])+1)**5

    for i in range(q):
        j = i+1
        while j < q:
            QBB[i, j] = y_train[int(WorkingSet[i, 0])]*y_train[int(WorkingSet[j, 0])]*Kernel(int(WorkingSet[i, 0]),int(WorkingSet[j, 0])) # (DotProduct(int(WorkingSet[i, 0]),int(WorkingSet[j, 0]))+1)**5 #(np.dot(x_train[int(WorkingSet[i, 0]), :],x_train[int(WorkingSet[j, 0]), :])+1)**5
            j = j + 1
    
    for i in range(q):
        j = i+1
        while j < q:
            QBB[j, i] = QBB[i, j]
            j = j + 1
            
    return QBB


def GenerateSwarm(SwarmSize, WorkingSet):
    Random = np.zeros(q)
    Signs = np.zeros(q)
    Size = 20
    print("Its me")
    
    Sum = 0
    for i in range (q):
        Signs[i] = y_train[int(WorkingSet[i, 0])]
        Sum = Sum + Signs[i]
    print(Signs)
    if Sum == q or Sum == -q:
        Size = WorkingSet[:, 1].max() - WorkingSet[:, 1].min()
    """
    if Signs[q] == 1:
        MarginPositive = C - WorkingSet[q, 1]
        MarginNegative = WorkingSet[q, 1]
    else:
        MarginPositive = WorkingSet[q, 1]
        MarginNegative = C - WorkingSet[q, 1]

    Sum = 0.0
    for i in range[q-1]:
        Sum = Sum + Signs[i]
    """
    for i in range(SwarmSize):
        Check = 1
        test = 0
        Counter = 0
        while Check != 3:
            Sum = 0.0
            Signs[q-1] = y_train[int(WorkingSet[q-1, 0])]
            for j in range(q-1):
                Random[j] = random.random()*Size - Size/2
                Signs[j] = y_train[int(WorkingSet[j, 0])]
                if WorkingSet[j, 1] + Random[j] > C or WorkingSet[j, 1] + Random[j] < 0:
                    Random[j] = -Random[j]
                Sum = Sum + y_train[int(WorkingSet[j, 0])]*Random[j]

            # Have to adjust random numbers to make the set feasible
            if -1*Signs[q-1]*Sum + WorkingSet[q-1, 1] < 0:
                Margin = -1*(Signs[q-1]*Sum + WorkingSet[q-1, 1])
                j = 0
                while j < q-1:
                    if Signs[j] == 1:
                        if WorkingSet[j, 1] + Random[j] + Margin < C and WorkingSet[j, 1] + Random[j] + Margin > 0:
                            Random[j] = Random[j] + Margin
                            j = q
                            Check = 3
                    else:
                        if WorkingSet[j, 1] + Random[j] - Margin > 0 and WorkingSet[j, 1] + Random[j] - Margin < C:
                            Random[j] = Random[j] - Margin
                            j = q
                            Check = 3
                    j = j + 1
                    test = 1

            if -1*Signs[q-1]*Sum + WorkingSet[q-1,1] > C:
                Margin = -1*Signs[q-1]*Sum + WorkingSet[q-1,1] - C
                j = 0
                while j < q-1:
                    if Signs[j] == 1:
                        if WorkingSet[j, 1] + Random[j] + Margin < C and WorkingSet[j, 1] + Random[j] + Margin > 0:
                            Random[j] = Random[j] + Margin
                            j = q
                            Check = 3
                    else:
                        if WorkingSet[j, 1] + Random[j] - Margin > 0 and WorkingSet[j, 1] + Random[j] - Margin < C:
                            Random[j] = Random[j] - Margin
                            j = q
                            Check = 3
                    j = j+1
                    test = 2

            Sum = 0.0
            for j in range(q-1):
                Sum = Sum + y_train[int(WorkingSet[j, 0])]*Random[j]
            Random[q-1] = -1*y_train[int(WorkingSet[q-1, 0])]*Sum
            
            for j in range(q):
                Swarm[i, j] = WorkingSet[j, 1] + Random[j]
                for j in range(q):
                    if Swarm[i, j] > C or Swarm[i, j] < 0:
                        Check = 1
                    else:
                        Check = 3
                # Check = 3
            """
                for j in range(q-1):
                    Sum = Sum + y_train[int(WorkingSet[j, 0])]*Random[j]
    
                Random[q-1] = y_train[int(WorkingSet[q-1, 0])]*-1*Sum
            
            if Check == 2:
                Random = np.zeros(q)
                Random[0] = random.random()*10
                if WorkingSet[0, 1] + Random[0] > C:
                    Random[0] = -Random[0]
                if Signs[0] + Signs[1] == 0:
                    Random[1] = Random[0]
                    if WorkingSet[1, 1] + Random[1] > C or WorkingSet[1, 1] + Random[1] < 0:
                        Random[1] = 0
                        Check = 4
                if Signs[0] + Signs[1] != 0:
                    Random[1] = -Random[0]
                    if WorkingSet[1, 1] + Random[1] > C or WorkingSet[1, 1] + Random[1] < 0:
                        Random[1] = 0
                        Check = 4
                test = 3

            if Check == 4:
                Random = np.zeros(q)
                Random[0] = random.random()*10
                if WorkingSet[0, 1] + Random[0] > C:
                    Random[0] = -Random[0]
                if Signs[0] + Signs[2] == 0:
                    Random[2] = Random[0]
                    if WorkingSet[2, 1] + Random[2] > C or WorkingSet[2, 1] + Random[2] < 0:
                        Random[2] = 0
                        Check = 5
                if Signs[0] + Signs[2] != 0:
                    Random[2] = -Random[0]
                    if WorkingSet[2, 1] + Random[2] > C or WorkingSet[2, 1] + Random[2] < 0:
                        Random[2] = 0
                        Check = 5
                test = 4

            if Check == 5:
                Random = np.zeros(q)
                Random[1] = random.random()*10
                if WorkingSet[1, 1] + Random[1] > C:
                    Random[1] = -Random[1]
                if Signs[1] + Signs[2] == 0:
                    Random[2] = Random[1]
                    if WorkingSet[2, 1] + Random[2] > C or WorkingSet[2, 1] + Random[2] < 0:
                        Random[2] = 0
                        Check = 1
                        test = 6
                if Signs[1] + Signs[2] != 0:
                    Random[2] = -Random[1]
                    if WorkingSet[2, 1] + Random[2] > C or WorkingSet[2, 1] + Random[2] < 0:
                        Random[2] = 0
                        Check = 1
                        test = 6

            if test != 6:
                
        """
        Sum = 0.0
        for j in range(q):
            Sum = Sum + Random[j]*Signs[j]
        if Sum != 0:
            print("Non linear problem")
            print(Random)
            print(Signs)
            print(WorkingSet)
            print(test)
            Check = 1
    print("Im done")
    # print(Swarm)
    if Swarm.min() < 0 or Swarm.max() > C:
        print("Generation Problem")
        print(test)
        # print(WorkingSet)
        print(Swarm)
    
    return Swarm


def DetermineObj(Swarm, QBB, qBN):
    ObjValue = np.zeros(len(Swarm), dtype='float64')
    One = np.ones(q)
    for i in range(len(Swarm)):
        ObjValue[i] = (np.matmul(np.transpose(Swarm[i, :]), One) - 0.5*(np.matmul(np.matmul(np.transpose(Swarm[i, :]), QBB),Swarm[i, :])) - np.matmul(np.transpose(Swarm[i, :]), qBN))
    return ObjValue

def DetermineObj1(Solution, QBB, qBN):
    ObjValue1 = 0.0
    One = np.ones(q)
    ObjValue1 = (np.matmul(np.transpose(Solution), One) - 0.5*(np.matmul(np.matmul(np.transpose(Solution), QBB),Solution)) - np.matmul(np.transpose(Solution), qBN))
    return ObjValue1


def GenerateV(Swarm, SwarmSize, WorkingSet, Margin):
    V = np.zeros((SwarmSize, q))
    Swarm1 = np.zeros((SwarmSize, q))
    Signs = np.zeros(q)
    for i in range(SwarmSize):
        Check = 1
        """
        while Check == 1:
            Sum = 0.0
            for j in range(q-1):
                V[i, j] = (random.random()*2 - 1)
                if Swarm[i, j] + V[i, j] > C:
                    V[i, j] = -V[i, j]
                Sum = Sum + y_train[int(WorkingSet[j, 0])]*V[i, j]
            V[i, q-1] = y_train[int(WorkingSet[q-1, 0])]*-1*Sum
            for j in range(q):
                Swarm1[i, j] = Swarm[i, j] + V[i, j]
            if Swarm1[i, q-1] > C or Swarm1[i, q-1] < 0:
                Check = 1
            else:
                Check = 2
        """
        while Check == 1:
            Sum = 0.0
            Signs[q-1] = y_train[int(WorkingSet[q-1, 0])]
            for j in range(q-1):
                V[i, j] = (random.random()*Margin - Margin/2)
                Signs[j] = y_train[int(WorkingSet[j, 0])]
                if Swarm[i, j] + V[i, j] > C or Swarm[i, j] + V[i, j] < 0:
                    V[i, j] = -V[i, j]
                Sum = Sum + y_train[int(WorkingSet[j, 0])]*V[i, j]

            # Have to adjust random numbers to make the set feasible
            if -1*Signs[q-1]*Sum + Swarm[i, q-1] < 0:
                Margin = (Signs[q-1]*Sum + Swarm[i, q-1])
                for j in range(q-1):
                    if Signs[j] == 1:
                        if Swarm[i, j] + V[i, j] + Margin < C:
                            V[i, j] = V[i, j] + Margin
                            j = q - 1
                    else:
                        if Swarm[i, j] + V[i, j] - Margin > 0:
                            V[i, j] = V[i, j] - Margin
                            j = q - 1

            if -1*Signs[q-1]*Sum + Swarm[i, q-1] > C:
                Margin = -1*Signs[q-1]*Sum + Swarm[i, q-1] - C
                for j in range(q-1):
                    if Signs[j] == 1:
                        if Swarm[i, j] + V[i, j] + Margin < C:
                            V[i, j] = V[i, j] + Margin
                            j = q - 1
                    else:
                        if Swarm[i, j] + V[i, j] - Margin > 0:
                            V[i, j] = V[i, j] - Margin
                            j = q - 1
            Sum = 0.0
            
            for j in range(q-1):
                Sum = Sum + y_train[int(WorkingSet[j, 0])]*V[i, j]

            V[i, q-1] = y_train[int(WorkingSet[q-1, 0])]*-1*Sum
            for j in range(q):
                Swarm1[i, j] = Swarm[i, j] + V[i, j]
            for j in range(q):
                if Swarm1[i, j] > C or Swarm1[i, j] < 0:
                    Check = 1
                else:
                    Check = 2
    # print(V)
    for i in range(SwarmSize):
        Sum = 0.0
        for j in range(q):
            Sum = Sum + V[i, j]*Signs[j]
        if Sum != 0:
            print("Non linear problem")
    return V


def CLPSO(WorkingSet, QBB, qBN):
    # Initialisation
    w = 0.7     # inertia weight
    c1 = 1.4
    c2 = 1.4
    rho = 1
    SwarmSize = 10
    Velocity = np.zeros((SwarmSize, q))
    MaxIterations = 100
    Swarm1 = np.zeros((SwarmSize, q))
    tracker = np.zeros(MaxIterations + 1)
    Swarm = GenerateSwarm(SwarmSize, WorkingSet)
    muSet = np.zeros(q)
    qBBCheck = np.zeros((1, q))
    Solution = np.zeros(q)
    Terminate = False
    k = 0
    GBest = np.zeros(q+1)
    PBest = np.zeros((SwarmSize, q+1))
    Margin = 2

    if Swarm.min() < 0 or Swarm.max() > C:
        print("Problem")

    ObjValue = DetermineObj(Swarm, QBB, qBN)

    PBest = np.column_stack([Swarm, ObjValue])
    SwarmValue = np.column_stack([Swarm, ObjValue])
    
    for i in range(q+1):
        GBest[i] = PBest[0, i]
    
    MeanVelocity = []
    
    while Terminate == False:
        
        # Determine GBest
        for i in range(SwarmSize):
            if PBest[i, q] > GBest[q]:
                for j in range(q+1):
                    GBest[j] = PBest[i, j]

        # UpdateVelocity
        r1 = random.random()
        r2 = random.random()
        V = GenerateV(Swarm, SwarmSize, WorkingSet, Margin)
        
        #Check = Swarm + V
        #if Check.min() < 0 or Check.max() > C:
            #print("Trouble")
        
        for i in range(SwarmSize):
            for j in range(q):
                if SwarmValue[i, q] == GBest[q]: # max(SwarmValue[:, q]):
                   Velocity[i, j] = rho*V[i, j] # PBest[i, j] - SwarmValue[i, j] + rho*V[i, j]
                else:
                    Velocity[i, j] = w*Velocity[i, j] + c1*r1*(PBest[i, j]-SwarmValue[i, j]) + c2*r2*(GBest[j]-SwarmValue[i, j])

        # Move particles to Swarm 1
        Sum = 0
        for i in range(SwarmSize):
            delta1 = 1
            delta = 1
            """
            for j in range(q):
                Swarm1[i, j] = Velocity[i, j] + Swarm[i, j]
                if Swarm1[i, j] < 0:
                    delta = (0-Swarm[i, j])/Velocity[i, j]
                    
                if Swarm1[i, j] > C:
                    delta = (C-Swarm[i, j])/Velocity[i, j]

                if delta < delta1:
                    delta1 = delta
            delta1 = max(0, delta1)
            """
            for j in range(q):
                Swarm[i][j] = delta1*Velocity[i][j] + Swarm[i][j]
            if delta1 == 0:
                Sum = Sum + 1
        """
        if Sum == SwarmSize:
            Margin = Margin/7
            V = GenerateV(Swarm, SwarmSize, WorkingSet, Margin)
            Swarm = Swarm + V
        """
        # Determine new objective value
        ObjValue = DetermineObj(Swarm, QBB, qBN)

        SwarmValue = np.column_stack([Swarm, ObjValue])
        # Determine PBest
        for i in range(SwarmSize):
            if PBest[i, q] < SwarmValue[i, q] and Swarm[i,:].min() >= 0 and Swarm[i,:].max() <= C:
                PBest[i, :] = SwarmValue[i, :]

        # Determine GBest
        for i in range(SwarmSize):
            if PBest[i, q] > GBest[q]:
                for j in range(q+1):
                    GBest[j] = PBest[i, j]
        tracker[k] = GBest[q]
        if k > 1 and tracker[k] < tracker[k-1]:
            print("Shit hit the fan")
            Terminate = True
        Count = 0
        for i in range(10):
            if tracker[k-i]-tracker[k-i-1] == 0:
                Count = Count + 1
        # if Count == 5:
        #     Random = np.random.randint(0, SwarmSize)
            """
            for i in range(q):
                Swarm = Swarm + V
                ObjValue = DetermineObj(Swarm, QBB, qBN)
                SwarmValue = np.column_stack([Swarm, ObjValue])
                # Determine PBest
                for i in range(SwarmSize):
                    if PBest[i, q] < SwarmValue[i, q]:
                        PBest[i, :] = SwarmValue[i, :]
                # Determine GBest
                for i in range(SwarmSize):
                    if PBest[i, q] > GBest[q]:
                        for j in range(q+1):
                            GBest[j] = PBest[i, j]
            """
        # print(Velocity[0,:])
        
        MeanVelocity.append(np.mean(np.absolute(Velocity)))
        
        # Condition check
        for i in range(q):
            Solution[i] = GBest[i]
        qBBCheck = np.matmul(QBB, Solution)
        Something = np.matmul(np.matmul(np.transpose(Solution), QBB),Solution)
        if Something <= 0:
            print("Problem")
        Sum = 0.0
        mu = 0.0
        for i in range(q):
            if Solution[i] > 0 and Solution[i] < C:
                mu = mu + y_train[int(WorkingSet[i, 0])]*(1-qBBCheck[i]-qBN[i])
                Sum = Sum + 1
                muSet[i] = y_train[int(WorkingSet[i, 0])]*(1-qBBCheck[i]-qBN[i])
        mu = mu/Sum
        Sum = 0.0
        for i in range(q):
            if Solution[i] <= 0 and qBBCheck[i] + qBN[i] + mu*y_train[int(WorkingSet[i, 0])] >= 1 - error:
                Sum = Sum + 1
            elif Solution[i] > 0 and Solution[i] < C and qBBCheck[i] + qBN[i] + mu*y_train[int(WorkingSet[i, 0])] > 1 - error and qBBCheck[i] + qBN[i] + mu*y_train[int(WorkingSet[i, 0])] < 1 + error:
                Sum = Sum + 1
            elif Solution[i] == C and qBBCheck[i] + qBN[i] + mu*y_train[int(WorkingSet[i, 0])] <= 1 + error:
                Sum = Sum + 1
        
        if Sum == q or k == MaxIterations:
            Terminate = True
            
        if GBest[q] - DetermineObj1(Solution, QBB, qBN) != 0:
            print("What the hell")
        # print(Sum)
        # print(k)
                
        k = k + 1
    for i in range(q):
        if GBest[i] < 1*10^(-10):
            GBest[i] = 0
    # print(Swarm)
    # plt.figure()
    # plt.plot(tracker)
    # plt.show()
    # print(tracker)
    # print(GBest)
    print(Sum)
    Yes = 0
    if Sum == 4.0:
        Yes = 1
    MeanVelocity = np.array(MeanVelocity)
    MeanVel = np.mean(MeanVelocity)
    print(MeanVel)
    return GBest, Yes


def decision_function(alphas, target, kernel, X_train, x_test, b):
    """Applies the SVM decision function to the input feature vectors in `x_test`."""

    result = (alphas * target) @ kernel(X_train, x_test) - b
    return result


def plot_decision_boundary(resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
        """Plots the model's decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`)."""

        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate the model over the entire space
        xrange = np.linspace(x_train[:,0].min(), x_train[:,0].max(), resolution)
        yrange = np.linspace(x_train[:,1].min(), x_train[:,1].max(), resolution)
        grid = [[decision_function(alpha, y_train,
                                   Kernel1, x_train,
                                   np.array([xr, yr]), b) for xr in xrange] for yr in yrange]
        grid = np.array(grid).reshape(len(xrange), len(yrange))

        # Plot decision contours using grid and
        # make a scatter plot of training data
        ax.contour(xrange, yrange, grid, levels=levels, linewidths=(1, 1, 1),
                   linestyles=('--', '-', '--'), colors=colors)
        ax.scatter(x_train[:,0], x_train[:,1],
                   c=y_train, cmap=plt.cm.viridis, lw=0, alpha=0.25)

        # Plot support vectors (non-zero alphas)
        # as circled points (linewidth > 0)
        mask = np.round(alpha, decimals=2) != 0.0
        ax.scatter(x_train[mask,0], x_train[mask,1],
                   c=y_train[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')

        return grid, ax

def Kernel1(x, y):
    """Calculate the Kernel value of x and y"""
    # Result = (np.dot(x, y.T)+1)**5
    Result = (x @ y.T + 1)**5
    # Sum = DotProduct(x, y)
    # Result = (Sum+1)**5
    
    return Result

"""
# The other test dataset
X_train, y_train = make_blobs(n_samples=1000, centers=2,
                        n_features=2, random_state=1)

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train, y_train)

x_train = x_train/20

y_train[y_train == 0] = -1
"""

# The self-generated sin-wave dataset
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

x_train = x_train/2.5
"""
"""
# Importing the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test1, y_train, y_test1 = train_test_split(x_train, y_train, test_size = 50/60, random_state = 0)

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


# Normalising to between 0 and 0.1
x_train /= 1500
x_test /= 1500
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
"""
# Importing the forest cover dataset
data =pd.read_csv("covtype.csv")
X = data.iloc[:, 0:53].values
y = data.iloc[:, 54].values

X = X.astype('float32')

# Feature scaling
Max = 0
for i in range(len(X[1])):
    Max = X[:,i].max()
    X[:, i] = X[:, i]/(Max*50)


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
# Importing the breast cancer dataset
data = pd.read_csv("breast_cancer_wisconsin_clean.csv")
X = data.iloc[:, 1:9].values
y = data.iloc[:, 10].values

X = X.astype('float32')

# Feature scaling
Max = 0
for i in range(len(X[1])):
    Max = X[:,i].max()
    X[:, i] = X[:, i]/(Max/2)


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
"""
Results = np.zeros((30,2))
for p in range(1):
    # Parameter intialisation
    SwarmSize = 10
    c1 = 1.4
    c2 = 1.4
    w = 0.7
    epsilon = 0.1
    error = 0.001
    q = 4
    C = 100
    Gamma = min(countX(y_train.tolist(), 1), countX(y_train.tolist(), -1))
    gamma = 1
    alpha = np.zeros(len(x_train))
    s = np.zeros(len(x_train))
    b1 = np.zeros(len(x_train))
    g = np.zeros(len(y_train))
    L = np.zeros((len(y_train), 3))
    WorkingSet = np.zeros((q, 3))
    qBN = np.zeros(q)
    QBB = np.zeros((q, q))
    Swarm = np.zeros((SwarmSize, q))
    WorkingSets = 0
    Optimise = True
    SVs = np.zeros(2*gamma)
    Difference = np.zeros(q)

    CountOptimal = 0

    start = time.time()
    iterations = 0
    Terminate = False
    WorkingSets += 1

    # Generate initial feasible solution
    alpha = GenerateInitialSolution()

    SupportVectorsInit = alpha.nonzero()

    # Calculate initial values of s
    s = GenerateS(s)

    # Caluculate initial value of b
    Sum = 0.0
    for i in range(len(SVs)):
        index = int(SVs[i])
        Sum = Sum + (y_train[index]-s[index])
    b = (1/(np.count_nonzero(alpha)))*Sum

    """
    for i in range(len(x_train)):
        b1[i] = y_train[i] - s[i]
    b = np.mean(b1)
    """

    # Let the loop begin for a working set
    while Terminate == False:
        iterations = iterations + 1

        # Calculate the value of g
        g = CalculateG(g, s)

        # Sort L in ascending order
        L = SortL(L, g)

        # Select the working set
        WorkingSet = SelectWorkingSet(L, q)
        print(WorkingSet)

        # Calculate the Hessian matrix QBB
        QBB = CalculateQBB(QBB, WorkingSet)

        # Calculate the vector qBN
        qBN = CalculateqBN(qBN, WorkingSet)

        # Determine the new GBest values - run optimisation
        Yes = 0
        GBest, Yes = CLPSO(WorkingSet, QBB, qBN)
        Counter = 0
        while Yes == 0 and Counter < 10:
            # print("Going again")
            Counter = Counter + 1
            print(Counter)
            GBest, Yes = CLPSO(WorkingSet, QBB, qBN)

        CountOptimal = CountOptimal + Yes

        # Update the values of alpha with the optimised values
        for i in range(q):
            WorkingSet[i, 1] = GBest[i]
            Difference[i] = GBest[i] - alpha[int(WorkingSet[i, 0])]
            alpha[int(WorkingSet[i, 0])] = GBest[i]

        for i in range(q):
            if GBest[i] > 0:
                SVs = np.append(SVs, WorkingSet[i, 0])
        Deletes = 0
        for i in range(len(SVs)):
            for j in range(q):
                if WorkingSet[j, 0] == SVs[i - Deletes]:
                    if WorkingSet[j, 1] <= 0:
                        SVs = np.delete(SVs, (i - Deletes), 0)
                        Deletes = Deletes + 1
        SVs = np.unique(SVs)

        # Update s
        s = UpdateS(s, Difference, WorkingSet)

        # Update b
        Sum = 0.0
        Sum1 = 0.0
        for i in range(len(y_train)):
            if alpha[i] > 0:
                Sum1 = Sum1 + 1
                Sum = Sum + (y_train[i]-s[i])
        b = (1/(Sum1))*Sum

        """
        for i in range(len(x_train)):
            b1[i] = y_train[i] - s[i]
        b = np.average(b1)
        """

        """
        Sum = 0.0
        for i in range(len(SVs)):
            Sum = Sum + y_train[int(SVs[i])] - s[int(SVs[i])]
        b = 1/(len(SVs))*(Sum)
        """

        Sum1 = 0
        Sum2 = 0
        Sum3 = 0
        Check1 = 0
        Check2 = 0
        Check3 = 0
        for i in range(len(L)):
            if alpha[i] > 0 and alpha[i] < C:
                Check1 = Check1 + 1
                if y_train[i]*(s[i]+b) > 1 - epsilon and y_train[i]*(s[i]+b) < 1 + epsilon:
                    Sum1 = Sum1 + 1
                #else:
                #    print(i)
                #    print(alpha[i])
                #    print(y_train[i]*(s[i]+b))
            if alpha[i] <= 0:
                Check2 = Check2 + 1
                if y_train[i]*(s[i]+b) > 1 - epsilon:
                    Sum2 = Sum2 + 1
                #else:
                #    print(i)
                #    print(alpha[i])
                #    print(y_train[i]*(s[i]+b))
            if alpha[i] >= C:
                Check3 = Check3 + 1
                if y_train[i]*(s[i]+b) < 1 + epsilon:
                    Sum3 = Sum3 + 1
                #else:
                #    print(i)
                #    print(alpha[i])
                #    print(y_train[i]*(s[i]+b))
        if (Sum1 + Sum2 + Sum3) >= len(alpha)*0.95:#  or iterations == 10000:
            Terminate = True
        print(Sum1 + Sum2 + Sum3)
        # if(len(alpha)-(Sum1 + Sum2 + Sum3)) < 2:
        #    Terminate = True
        print(Check1 - Sum1)
        print(Check2 - Sum2)
        print(Check3 - Sum3)
        # print(np.count_nonzero(alpha))
        # print(iterations)
        Sum = 0.0
        for i in range(len(x_train)):
            Sum = Sum + y_train[i]*alpha[i]
        print(Sum)

    print("This is how long it took")
    print(time.time() - start)

    print(np.count_nonzero(alpha))
    print(max(alpha))
    print(min(alpha))
    print(iterations)

    print(alpha.nonzero())
    Results[p, 0] = time.time() - start
    Results[p, 1] = iterations

# fig, ax = plt.subplots()
# grid, ax = plot_decision_boundary()

Sum = 0.0
for i in range(len(x_train)):
    Sum = Sum + y_train[i]*alpha[i]
print(Sum)

SupportVectors = alpha.nonzero()
print("This is the total number of iterations:", iterations)
print("This is the total number of optimal solution iterations:", CountOptimal)


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
        
Sum = 0
for i in range(len(x_train)):
    if alpha[i] < 0:
        Sum = Sum + 1
print(Sum)

PositiveT = 0
NegativeT = 0
PositiveF = 0
NegativeF = 0
for j in range(len(x_test)):
    Sum = 0.0
    for i in range(len(SVs)):
        Sum = Sum + y_train[int(SVs[i])]*alpha[int(SVs[i])]*KernelTest(j, int(SVs[i]))
    Classification = Sum + b
    if Classification > 0 and y_test[j] == 1:
        PositiveT = PositiveT + 1
    elif Classification > 0 and y_test[j] == -1:
        PositiveF = PositiveF + 1
    elif Classification < 0 and y_test[j] == -1:
        NegativeT = NegativeT + 1
    else:
        NegativeF = NegativeF + 1

print(PositiveT,PositiveF, NegativeT, NegativeF)
