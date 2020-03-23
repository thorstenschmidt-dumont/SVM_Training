#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:25:06 2019

@author: thorsten
"""

import numpy as np
import random
import functools
import math
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import time

#Function to perform gauss jordan reduction


def gauss_jordan(m, eps = 1.0/(10**10)):
  """Puts given matrix (2D array) into the Reduced Row Echelon Form.
     Returns True if successful, False if 'm' is singular.
     NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
     Written by Jarno Elonen in April 2005, released into Public Domain"""
  (h, w) = (len(m), len(m[0]))
  for y in range(0,h):
    maxrow = y
    for y2 in range(y+1, h):    # Find max pivot
      if abs(m[y2][y]) > abs(m[maxrow][y]):
        maxrow = y2
    (m[y], m[maxrow]) = (m[maxrow], m[y])
    if abs(m[y][y]) <= eps:     # Singular?
      return False
    for y2 in range(y+1, h):    # Eliminate column y
      c = m[y2][y] / m[y][y]
      for x in range(y, w):
        m[y2][x] -= m[y][x] * c
  for y in range(h-1, 0-1, -1): # Backsubstitute
    c  = m[y][y]
    for y2 in range(0,y):
      for x in range(w-1, y-1, -1):
        m[y2][x] -=  m[y][x] * m[y2][y] / c
    m[y][y] /= c
    for x in range(h, w):       # Normalize row y
      m[y][x] /= c
  return True

# Function to determine objective function values


def DetermineObj(Swarm):
    ObjValue = np.zeros(len(Swarm), dtype='float64')

    # Objective function 1
    
    for i in range(len(Swarm)):
        Particle = Swarm[i]
        Particle = np.array(Particle)
        ObjValue[i] = sum(np.square(Particle))
    

    # Objective function 2
    """
    for z in range(len(Swarm)):
        Sum = np.zeros(1, dtype='float64')
        for i in range(len(Swarm[0])):
            for j in range(len(Swarm[0])):
                Sum = Sum + (math.exp(-(Swarm[z][i]-Swarm[z][j])**2)*Swarm[z][i]*Swarm[z][j])
        ObjValue[z] = Sum + sum(Swarm[z, :])
    """
    # Objective function 3
    """
    for z in range(len(Swarm)):
        Sum = np.zeros(1,dtype = 'float64')
        for i in range(len(Swarm[0])-1):
            Sum = Sum + (100*(Swarm[z][i+1]-Swarm[z][i]**2)**2+(1-Swarm[z][i])**2)
        ObjValue[z] = Sum
    """
    return ObjValue

# Generate initial swarms


def GenerateInitialSolution(SwarmSize):
    aug = np.concatenate((A, b), 1)

    aug = aug.tolist()

    gauss_jordan(aug)
    aug = np.array(aug)

    RandomP = np.zeros(len(A[0])-len(A), dtype='float64')

    Swarm = np.zeros((SwarmSize, len(A[0])), dtype='float64')

    for i in range(SwarmSize):
        Sum = np.zeros(len(A), dtype='float64')
        for j in range(len(A[0])-len(A)):
            RandomP[j] = random.randrange(-100, 100)
            Swarm[i][j+len(A)] = RandomP[j]
        for k in range(len(A)):
            for j in range(len(A[0])-len(A)):
                Sum[k] = Sum[k] + aug[k][j+len(A[0])-len(A)]*RandomP[j]
        for k in range(len(A)):
            Swarm[i][k] = aug[k][len(A[0])] - Sum[k]
    return Swarm

# Generate initial vector v


def GenerateV(SwarmSize):
    aug = np.concatenate((A, c), 1)

    aug = aug.tolist()

    gauss_jordan(aug)
    aug = np.array(aug)

    RandomP = np.zeros(len(A[0])-len(A), dtype='float64')

    V = np.zeros((SwarmSize, len(A[0])), dtype='float64')

    for i in range(SwarmSize):
        Sum = np.zeros(len(A), dtype='float64')
        for j in range(len(A[0])-len(A)):
            RandomP[j] = random.random()*2 - 1
            V[i][j+len(A)] = RandomP[j]
        for k in range(len(A)):
            for j in range(len(A[0])-len(A)):
                Sum[k] = Sum[k] + aug[k][j+len(A[0])-len(A)]*RandomP[j]
        for k in range(len(A)):
            V[i][k] = aug[k][len(A[0])] - Sum[k]
    return V

# Problem parameters

A = np.array([[0.0,-3.0,-1.0,0.0,0.0,2.0,-6.0,0.0,-4.0,-2.0],
              [-1.0,-3.0,-1.0,0.0,0.0,0.0,-5.0,-1.0,-7.0,-2.0],
              [0.0,0.0,1.0,0.0,0.0,1.0,3.0,0.0,-2.0,2.0],
              [2.0,6.0,2.0,2.0,0.0,0.0,4.0,6.0,16.0,4.0],
              [-1.0,-6.0,-1.0,-2.0,-2.0,3.0,-6.0,-5.0,-13.0,-4.0]])

b = np.array([[3.0],
              [0.0],
              [9.0],
              [-16.0],
              [30.0]])

c = np.array([[0.0],
              [0.0],
              [0.0],
              [0.0],
              [0.0]])

"""
A = np.array([[1.0,2.0,3.0,4.0,5.0],
        [5.0,4.0,3.0,2.0,1.0],
        [7.0,6.0,4.0,6.0,7.0]])

b = np.array([[143.0],
              [91.0],
              [236.0]])

"""


def CLPSO():
    # Initialisation
    w = 0.7     # inertia weight
    c1 = 1.4
    c2 = 1.4
    rho = 1
    SwarmSize = 20
    Velocity = np.zeros((SwarmSize, len(A[0])))
    MaxIterations = 250

    Swarm = GenerateInitialSolution(SwarmSize)

    ObjValue = DetermineObj(Swarm)

    PBest = np.column_stack([Swarm, ObjValue])
    SwarmValue = np.column_stack([Swarm, ObjValue])

    GBest = PBest[0, :]

    for k in range(MaxIterations):
        # Determine GBest
        for i in range(SwarmSize):
            if PBest[i, 10] <= GBest[10]:
                GBest = PBest[i, :]
        # UpdateVelocity
        r1 = random.random()
        r2 = random.random()
        V = GenerateV(SwarmSize)
        for i in range(SwarmSize):
            for j in range(len(A[0])):
                if SwarmValue[i, 10] == GBest[10]: # min(SwarmValue[:, 10]):
                    Velocity[i, j] = rho*V[i, j] # PBest[i][j] - SwarmValue[i, j] + rho*V[i, j]
                else:
                    Velocity[i, j] = w*Velocity[i, j] + c1*r1*(PBest[i, j]-SwarmValue[i, j]) + c2*r2*(GBest[j]-SwarmValue[i, j])
        # Move particles
        for i in range(SwarmSize):
            for j in range(len(A[0])):
                Swarm[i][j] = Velocity[i][j] + Swarm[i][j]
        # Determine new objective value
        ObjValue = DetermineObj(Swarm)

        SwarmValue = np.column_stack([Swarm, ObjValue])
        # Determine PBest
        for i in range(SwarmSize):
            if PBest[i, 10] >= ObjValue[i]:
                PBest[i, :] = SwarmValue[i, :]
        ++k
    return GBest[10]


# Experimental design
def ResultsGen(i):
    Results[i] = CLPSO()
    return Results[i]


Results = np.zeros(100)

#Result = CLPSO()
#print(Result)

start = time.time()
pool = Pool(cpu_count())
Results = pool.map(ResultsGen, range(100))
end = time.time()
print(end-start)
print(np.mean(Results))
print(max(Results))
print(min(Results))
print(np.std(Results))

"""
Sum = np.zeros(SwarmSize)
for i in range(len(A[0])):
    Sum[0] = Sum[0] + Swarm[2][i]*A[4][i]

print(Sum)
"""
