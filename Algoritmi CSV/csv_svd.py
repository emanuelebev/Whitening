# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:52:56 2021

@author: emanu
"""

import sys
import numpy as np
import matplotlib as mpl
if sys.platform == 'darwin': mpl.use('TkAgg')
import csv
import time

from scipy import linalg

start_time = time.time()

# STEP 0: Load data
#file = open("data/pd_755_cols.csv")
#file = open("data/HAR/test.csv")
#x = np.loadtxt(file, delimiter=",", skiprows=1)

with open("Data/HAR/train.csv") as f:
    ncols = len(f.readline().split(','))
x = np.loadtxt("Data/HAR/train.csv", delimiter=',', skiprows=1, usecols=range(2,ncols))

# STEP 1a: Implement PCA to obtain the rotation matrix, U, which is
# the eigenbases sigma.

sigma = x.dot(x.T) / x.shape[1]
U, S, Vh = linalg.svd(sigma)

# STEP 1b: Compute xRot, the projection on to the eigenbasis

xRot = U.T.dot(x)


# STEP 2: Reduce the number of dimensions from 2 to 1

k = 1
xRot = U[:,0:k].T.dot(x)
xHat = U[:,0:k].dot(xRot)


# STEP 3: PCA Whitening

epsilon = 1e-5
xPCAWhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T).dot(x)


# STEP 4: ZCA Whitening

xZCAWhite = U.dot(xPCAWhite)

with open("data/Result/result_PCA.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(xPCAWhite)
with open("data/Result/result_ZCA.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(xZCAWhite)
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- Time: %s seconds ---" % (time.time() - start_time))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    