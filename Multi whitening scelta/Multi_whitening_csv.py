# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:32:06 2021

@author: emanu
"""

import numpy as np
import time
import sys
import matplotlib as mpl
if sys.platform == 'darwin': mpl.use('TkAgg')
import csv

start_time = time.time()

def whiten(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None
    
    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        #data centered and scaled before the analysis
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)

# file = open("data/HAR/test.csv")
# y = np.loadtxt(file, delimiter=",", skiprows=1)

with open("Data/HAR/train.csv") as f:
    ncols = len(f.readline().split(','))
y = np.loadtxt("Data/HAR/train.csv", delimiter=',', skiprows=1, usecols=range(2,ncols))


x = whiten(y, "zca_cor")
with open("data/Result/result_whiten.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(x)
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- Time: %s seconds ---" % (time.time() - start_time))



