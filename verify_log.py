#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:44:59 2021

@author: emanuelebevacqua
"""

# Import libraries
import numpy as np
from scipy import linalg
import csv
import time
import json
import random
from random import randrange


start_time = time.time()
 
json_data = []

#fatto
# json_file = "data/electricity_nips/test/data.json"      # 4000 cols, 2590 rows
# json_file = "data/electricity_nips/train/data.json"      # 5833 cols, 370 rows
# json_file = "data/exchange_rate_nips/test/test.json"    # 6221 cols, 40 rows
# json_file = "data/exchange_rate_nips/train/train.json"    # 6071 cols, 8 rows
# json_file = "data/solar_nips/test/test.json"    # 7177 cols, 959 rows
# json_file = "data/solar_nips/train/train.json"    # 7009 cols, 137 rows
# json_file = "data/taxi_30min/train/train.json"    # 1488 cols, 1214 rows

#mancano
# json_file = "data/traffic_nips/test/data.json"    # 4000 cols, 6741 rows
# json_file = "data/traffic_nips/train/data.json"    # 4001 cols, 963 rows

#fatto
# json_file = "data/wiki-rolling_nips/train/train.json"    # 792 cols, 9535 rows
# json_file = "data/Liquid_water/test/data.json"     # 1008 cols, 414 rows
# json_file = "data/Liquid_water/train/data.json"     # 960 cols, 414 rows
# json_file = "data/Liquid_oil/test/data.json"     # 9933 cols, 4227 rows
# json_file = "data/Liquid_oil/train/data.json"     # 9919 cols, 4227 rows

#mancano
json_file = "data/Liquid_gas/test/data.json"     # 2610 cols, 359 rows
# json_file = "data/Liquid_gas/train/data.json"     # 2597 cols, 359 rows

#fatto
# json_file = "data/pressure_sensor/data.json"     # 1871 cols, 3998 rows
# json_file = "data/light_sensor/data.json"     # 2659 cols, 12805 rows
# json_file = "data/accelerometer/test/data.json"     # 874 cols, 14833 rows
# json_file = "data/accelerometer/train/data.json"     # 866 cols, 10163 rows
# json_file = "data/ultrasonic_sensor/test/data.json"     # 265 cols, 7405 rows
# json_file = "data/ultrasonic_sensor/train/data.json"     # 259 cols, 5752 rows

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



# Zero center data
xc = x - np.mean(x, axis=0)
xc = xc.T

# Calculate Covariance matrix
# Note: 'rowvar=True' because each row is considered as a feature
# Note: 'bias=True' to divide the sum of squared variances by 'n' instead of 'n-1'
xcov = np.cov(xc, rowvar=True, bias=True)
 
# Calculate Eigenvalues and Eigenvectors
w, U = linalg.eig(xcov) # .eigh()
# w = w.real.round(4)
# v = v.real.round(4)
# Note: Use w.real.round(4) to (1) remove 'j' notation to real, (2) round to '4' significant digits


Ut = U.T


# Calculate inverse square root of Eigenvalues
# Optional: Add '.1e5' to avoid division errors if needed
# Create a diagonal matrix
# Diagonal matrix for inverse square root of Eigenvalues
diagw = np.diag(1/((w+.1e-5)**0.5)) # np.diag(1/(w**0.5)) or np.diag(1/((w+.1e-5)**0.5))
#diagw = diagw.real.round(4) #convert to real and round off

d = ''

            
# Calculate Rotation (optional)
# Note: To see how data can be rotated
xrot = np.dot(U, xc)
 
# Whitening transform using PCA (Principal Component Analysis)
wpca = np.dot(np.dot(diagw, U.T), xc)
PCA_time = time.time() - start_time
 
# Whitening transform using ZCA (Zero Component Analysis)
wzca = np.dot(np.dot(np.dot(U, diagw), U.T), xc)
ZCA_time = time.time() - start_time




W = np.dot(diagw, Ut)
Sig = xcov
Wt = W.T
Eq2 = np.dot(np.dot(W, Sig), Wt)
Eq2 = Eq2.real.round(0)

with open("data/Result/Verify/Eq2.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Eq2)

errorEq2 = 0
df=open('data/Result/Verify/checkEq2.txt','w')
for i in range(len(Eq2)):
    for j in range(len(Eq2[i])):
        if (Eq2[i][j] != 1.0 and i == j):
            df.write(str(i))
            df.write("   ")
            df.write(str(j))
            df.write("   ")
            df.write(str(Eq2[i][j]))
            df.write("\n")
            errorEq2 += 1
            
celleEq2 = (len(Eq2)*len(Eq2[0]))
percErrEq2 = 100 - (((celleEq2 - errorEq2)*100)/celleEq2)



appEq3 = np.dot(Sig, np.dot(Wt, W))
Eq3 = np.dot(W, appEq3)
Eq3 = Eq3.real.round(5)
W = W.real.round(5)

with open("data/Result/Verify/Eq3.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Eq3)
with open("data/Result/Verify/W.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(W)

errorEq3 = 0
sumErrorEq3 = 0
df=open('data/Result/Verify/checkEq3.txt','w')
for i in range(len(Eq3)):
    for j in range(len(Eq3[i])):
        if(Eq3[i][j] != W[i][j]):
            df.write(str(i))
            df.write("   ")
            df.write(str(j))
            df.write("   ")
            df.write(str(Eq3[i][j]))
            df.write("   ")
            df.write(str(W[i][j]))
            df.write("\n")
            sumErrorEq3 += abs(Eq3[i][j] - W[i][j])
            errorEq3+=1

            
            

Eq4 = np.dot(Wt, W).real.round(1)
# SigInv = np.linalg.inv(Sig).real.round(1)
SigInv = np.dot(np.dot(U, diagw), Ut)

with open("data/Result/Verify/Eq4.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Eq4)

with open("data/Result/Verify/SigInv.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(SigInv)


sumErrorEq4 = 0
errorEq4 = 0
df=open('data/Result/Verify/checkEq4.txt','w')
for i in range(len(Eq4)):
    for j in range(len(Eq4[i])):
        if(Eq4[i][j] != SigInv[i][j]):
            df.write(str(i))
            df.write("   ")
            df.write(str(j))
            df.write("   ")
            df.write(str(Eq4[i][j]))
            df.write("   ")
            df.write(str(SigInv[i][j]))
            df.write("\n")
            sumErrorEq4 += (Eq4[i][j] - SigInv[i][j])
            errorEq4 += 1
            




# print("--- Columns: %d ---" % len(x[0]))
# print("--- Rows: %d ---" % len(x))
# print("--- Time: %s seconds ---" % (time.time() - start_time))

print("File:%s " % json_file)
print("Columns:%d " % len(x[0]))
print("Rows:%d " % len(x))
print("Time:%s " % (time.time() - start_time))
print("errorEq2:%f "% errorEq2)
print("percentageErrorEq2:%f%% " % percErrEq2)
print("errorEq3:%f "% errorEq3)
print("sumErrorEq3:%f "% sumErrorEq3)
print("meanErrorEq3:%f " % (sumErrorEq3 / errorEq3))
print("meanErrorEq4:%f " % ((sumErrorEq4 / errorEq4)/1))


log = open('data/Result/Verify/log.txt','w')
log.write("File:%s " % json_file)
log.write("Columns:%d " % len(x[0]))
log.write("Rows:%d " % len(x))
log.write("Time:%s " % (time.time() - start_time))
log.write("errorEq2:%f "% errorEq2)
log.write("percentageErrorEq2:%f%% " % percErrEq2)
log.write("errorEq3:%f "% errorEq3)
log.write("sumErrorEq3:%f "% sumErrorEq3)
log.write("meanErrorEq3:%f " % (sumErrorEq3 / errorEq3))
log.write("meanErrorEq4:%f " % ((sumErrorEq4 / errorEq4)/10000))








