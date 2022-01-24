#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:05:04 2021

@author: emanuelebevacqua
"""

import numpy as np
from scipy import linalg
import csv
import time
import json
import pandas as pd

import psutil



start_time = time.time()
 
# Create data
#np.random.seed(1)
#mu = [0,0]
#sigma = [[6,5], [5,6]]
# n = 1000
# x = np.random.multivariate_normal(mu, sigma, size=n)
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
# json_file = "data/SWaT_oil/test/data.json"     # 1008 cols, 414 rows
# json_file = "data/SWaT_oil/train/data.json"     # 960 cols, 414 rows
# json_file = "data/SWaT_gas/test/data.json"     # 9933 cols, 4227 rows
# json_file = "data/SWaT_gas/train/data.json"     # 9919 cols, 4227 rows
# json_file = "data/SWaT_water/test/data.json"     # 2610 cols, 359 rows
# json_file = "data/SWaT_water/train/data.json"     # 2597 cols, 359 rows
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
x = np.array(array)
 
# Zero center data
xc = x - np.mean(x, axis=0)
xc = xc.T
 
# Calculate Covariance matrix
# Note: 'rowvar=True' because each row is considered as a feature
# Note: 'bias=True' to divide the sum of squared variances by 'n' instead of 'n-1'
xcov = np.cov(xc, rowvar=True, bias=True)
 
# Calculate Eigenvalues and Eigenvectors
w, v = linalg.eig(xcov) # .eigh()
w = w.real.round(4)
v = v.real.round(4)
correl = np.array(np.dot(np.dot(v, w), v.T))
correl = correl.real.round(4)
# Note: Use w.real.round(4) to (1) remove 'j' notation to real, (2) round to '4' significant digits
np.savetxt('data/Result/Eigenvalues/Eigenvalues.txt', w, delimiter=" ", fmt="%s") 
with open("data/Result/Eigenvalues/Eigenvectors.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(v)
with open("data/Result/Eigenvalues/Correlation_matrix.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(v)
    
df = pd.DataFrame(x)
corr = np.array(df.corr()).real.round(4)
with open("data/Result/Eigenvalues/Correlation_matrix.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(corr)
 
# Calculate inverse square root of Eigenvalues
# Optional: Add '.1e5' to avoid division errors if needed
# Create a diagonal matrix
diagw = np.diag(1/((w+.1e-5)**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
diagw = diagw.real.round(4) #convert to real and round off
with open("data/Result/Eigenvalues/diagonal_matrix.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(diagw)
            
# Calculate Rotation (optional)
# Note: To see how data can be rotated
xrot = np.dot(v, xc)
 
# Whitening transform using PCA (Principal Component Analysis)
wpca = np.dot(np.dot(diagw, v.T), xc)
wpca = wpca.real.round(4)
PCA_time = time.time() - start_time

df_pca = pd.DataFrame(x)
corr_pca = np.array(df_pca.corr()).real.round(4)
with open("data/Result/Eigenvalues/Correlation_matrix_pca.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(corr_pca)
 
# Whitening transform using ZCA (Zero Component Analysis)
wzca = np.dot(np.dot(np.dot(v, diagw), v.T), xc)
wzca = wzca.real.round(4)
ZCA_time = time.time() - start_time

with open("data/Result/Eigenvalues/PCA_result.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(wpca)
with open("data/Result/Eigenvalues/ZCA_result.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(wzca)
print("--- Dataset: %s ---" % json_file)
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- PCA time: %s seconds ---" % (PCA_time))
print("--- ZCA time: %s seconds ---" % (ZCA_time))
# gives a single float value
psutil.cpu_percent()
# gives an object with many fields
psutil.virtual_memory()
# you can convert that object to a dictionary 
dict(psutil.virtual_memory()._asdict())
# you can have the percentage of used RAM
print(psutil.virtual_memory().percent)






