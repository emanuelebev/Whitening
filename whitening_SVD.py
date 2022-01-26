# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:25:06 2021

@author: emanuelebevacqua
"""

import sys
import numpy as np
import matplotlib as mpl
if sys.platform == 'darwin': mpl.use('TkAgg')
import csv
import time
import json

from scipy import linalg

start_time = time.time()

# STEP 0: Load data
#file = open("data/pd_755_cols.csv")
#file = open("data/HAR/test.csv")
#x = np.loadtxt(file, delimiter=",", skiprows=1)

# with open("Data/HAR/train.csv") as f:
#     ncols = len(f.readline().split(','))
#x = np.loadtxt("Data/HAR/train.csv", delimiter=',', skiprows=1, usecols=range(2,ncols))

# Open the JSON file & load its data
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
            tar.append(1.0)
    array.append(np.array(tar))
x = np.array(array)

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
xPCAWhite = xPCAWhite.real.round(4)
PCA_time = time.time() - start_time


# STEP 4: ZCA Whitening

xZCAWhite = U.dot(xPCAWhite)
xZCAWhite = xZCAWhite.real.round(4)
ZCA_time = time.time() - start_time

with open("data/Result/SVD/PCA_result.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(xPCAWhite)
with open("data/Result/SVD/ZCA_result.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(xZCAWhite)
print("--- Dataset: %s ---" % json_file)
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- PCA time: %s seconds ---" % (PCA_time))
print("--- ZCA time: %s seconds ---" % (ZCA_time))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    