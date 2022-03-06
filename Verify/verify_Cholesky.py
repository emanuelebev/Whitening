#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 11:19:49 2021

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

def whiten(X):
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    U, Lambda, _ = np.linalg.svd(Sigma)
    W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    return np.dot(X_centered, W.T), W, Sigma


# Create data
json_data = []
json_file = "data/tab_json.json"
file = open(json_file)
for line in file:
 	json_line = json.loads(line)
 	json_data.append(json_line)
array = []
for i in range(len(json_data)):
    array.append(np.array(json_data[i]["target"]))
y = np.array(array)

# array = []
# for i in range(11):
#     app=[]
#     for j in range(10):
#         num = round(random.uniform(1,10), 4)
#         # num = randrange(10)
#         app.append(num)
#     array.append(np.array(app))
# y = np.array(array)

with open("data/Result/Cholesky/X.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(y)

# L is the unique lower triangular matrix with positive diagonal values
x, Lt, Sig = whiten(y)
cho_time = time.time() - start_time

with open("data/Result/Cholesky/L.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Lt.T.real.round(4))
with open("data/Result/Cholesky/L^T.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Lt.real.round(4))
with open("data/Result/Cholesky/Sigma.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Sig.real.round(4))

Identity = np.dot(np.dot(Lt, Sig), Lt.T)
with open("data/Result/Cholesky/L^T Sigma L.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Identity.round(4))

cov_Cho = np.cov(x, rowvar=True, bias=True)
with open("data/Result/Cholesky/cov_Cho.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(cov_Cho.real.round(4))
    
with open("data/Result/Cholesky/Cholesky_result.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(x.real.round(4))

print("--- Dataset: %s ---" % json_file)
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- Time: %s seconds ---" % (cho_time))







