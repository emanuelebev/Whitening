#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:18:45 2021

@author: emanuelebevacqua
"""

import numpy as np
import csv
import time
import json

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

json_data = []
#json_file = "data/electricity_nips/test/data.json"      # 4000 cols, 2590 rows
#json_file = "data/electricity_nips/train/data.json"      # 5833 cols, 370 rows
#json_file = "data/exchange_rate_nips/test/test.json"    # 6221 cols, 40 rows
#json_file = "data/exchange_rate_nips/train/train.json"    # 6071 cols, 8 rows
#json_file = "data/solar_nips/test/test.json"    # 7177 cols, 959 rows
#json_file = "data/solar_nips/train/train.json"    # 7009 cols, 137 rows
#json_file = "data/taxi_30min/train/train.json"    # 1488 cols, 1214 rows
#json_file = "data/traffic_nips/test/data.json"    # 4000 cols, 6741 rows
#json_file = "data/traffic_nips/train/data.json"    # 4001 cols, 963 rows
json_file = "data/wiki-rolling_nips/train/train.json"    # 792 cols, 9535 rows

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
f = np.array(array)
y = f - np.mean(f, axis=0)

start_w = time.time()
x, w = whiten(y)
print("--- Eigenvalues time: %s seconds ---" % (time.time() - start_w))

start_svd = time.time()
svd = svd_whiten(y)
print("--- SVD time: %s seconds ---" % (time.time() - start_svd))

with open("data/Result/Simple/eig_whiten.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(x)
with open("data/Result/Simple/svd_whiten.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(svd)
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- Total time: %s seconds ---" % (time.time() - start_time))