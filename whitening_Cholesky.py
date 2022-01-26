#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:19:43 2021

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
    return np.dot(X_centered, W.T)

json_data = []
# json_file = "data/electricity_nips/test/data.json"      # 4000 cols, 2590 rows
# json_file = "data/electricity_nips/train/data.json"      # 5833 cols, 370 rows
# json_file = "data/exchange_rate_nips/test/test.json"    # 6221 cols, 40 rows
# json_file = "data/exchange_rate_nips/train/train.json"    # 6071 cols, 8 rows
# json_file = "data/solar_nips/test/test.json"    # 7177 cols, 959 rows
# json_file = "data/solar_nips/train/train.json"    # 7009 cols, 137 rows
# json_file = "data/taxi_30min/train/train.json"    # 1488 cols, 1214 rows
# json_file = "data/traffic_nips/test/data.json"    # 4000 cols, 6741 rows
# json_file = "data/traffic_nips/train/data.json"    # 4001 cols, 963 rows
# json_file = "data/wiki-rolling_nips/train/train.json"    # 792 cols, 9535 rows
# json_file = "data/Liquid_oil/test/data.json"     # 1008 cols, 414 rows
# json_file = "data/Liquid_oil/train/data.json"     # 960 cols, 414 rows
# json_file = "data/Liquid_gas/test/data.json"     # 9933 cols, 4227 rows
# json_file = "data/Liquid_gas/train/data.json"     # 9919 cols, 4227 rows
# json_file = "data/Liquid_water/test/data.json"     # 2610 cols, 359 rows
# json_file = "data/Liquid_water/train/data.json"     # 2597 cols, 359 rows
# json_file = "data/pressure_sensor/data.json"     # 1871 cols, 3998 rows
# json_file = "data/light_sensor/data.json"     # 2659 cols, 12805 rows
# json_file = "data/accelerometer/test/data.json"     # 874 cols, 14833 rows
# json_file = "data/accelerometer/train/data.json"     # 866 cols, 10163 rows
# json_file = "data/ultrasonic_sensor/test/data.json"     # 265 cols, 7405 rows
json_file = "data/ultrasonic_sensor/train/data.json"     # 259 cols, 5752 rows


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

x = whiten(y)
cho_time = time.time() - start_time

# with open("data/Result/Cholesky/SVD_whiten.csv", 'w', newline='') as csvfile: 
#     csvwriter = csv.writer(csvfile) 
#     csvwriter.writerows(x)
print("--- Dataset: %s ---" % json_file)
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- Time: %s seconds ---" % (cho_time))