#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:14:11 2021

@author: emanuelebevacqua
"""

import numpy as np
import csv
import time

start_time = time.time()

def whiten(X,fudge=1e-5):

   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = np.dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d, V = np.linalg.eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1. / np.sqrt(d+fudge))

   # whitening matrix
   W = np.dot(np.dot(V, D), V.T)

   # multiply by the whitening matrix
   X_white = np.dot(X, W)

   return X_white, W

def svd_whiten(X):

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white


file = open("data/HAR/test.csv")
f = np.loadtxt(file, delimiter=",", skiprows=1)
xc = f - np.mean(f, axis=0)
y = xc.T

start_w = time.time()
x, w = whiten(y)
print("--- Eigenvalues time: %s seconds ---" % (time.time() - start_w))

start_svd = time.time()
svd = svd_whiten(y)
print("--- SVD time: %s seconds ---" % (time.time() - start_svd))

with open("data/Result/SVD/whiten.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(x)
with open("data/Result/SVD/svd_whiten.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(svd)
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- Total time: %s seconds ---" % (time.time() - start_time))


