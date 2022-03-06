#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:09:52 2021

@author: emanuelebevacqua
"""

import numpy as np
import time
import sys
import matplotlib as mpl
if sys.platform == 'darwin': mpl.use('TkAgg')
import csv
import json

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

json_data = []
#json_file = "data/electricity_nips/test/data.json"      # 4000 cols, 2590 rows
json_file = "data/electricity_nips/train/data.json"      # 5833 cols, 370 rows
#json_file = "data/exchange_rate_nips/test/test.json"    # 6221 cols, 40 rows
#json_file = "data/exchange_rate_nips/train/train.json"    # 6071 cols, 8 rows
#json_file = "data/solar_nips/test/test.json"    # 7177 cols, 959 rows
#json_file = "data/solar_nips/train/train.json"    # 7009 cols, 137 rows
#json_file = "data/taxi_30min/train/train.json"    # 1488 cols, 1214 rows
#json_file = "data/traffic_nips/test/data.json"    # 4000 cols, 6741 rows
#json_file = "data/traffic_nips/train/data.json"    # 4001 cols, 963 rows
#json_file = "data/wiki-rolling_nips/train/train.json"    # 792 cols, 9535 rows


file = open(json_file)
for line in file:
	json_line = json.loads(line)
	json_data.append(json_line)

max=0
for j in range(len(json_data)):
    lun = len(json_data[j]["target"])
    if(lun > max):
        max = lun

array = []
for i in range(len(json_data)):
    tar = json_data[i]["target"]
    l = len(tar)
    if(l < max):
        for app in range(max - l):
            tar.append(0.0)
    array.append(np.array(tar))
y = np.array(array)

x = whiten(y, "cholesky")
with open("data/Result/Multiple/result_whiten.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(x)
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- Time: %s seconds ---" % (time.time() - start_time))