#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:29:51 2021

@author: emanuelebevacqua
"""

# Import libraries
import numpy as np
from scipy import linalg
import csv
import time
import pandas as pd

start_time = time.time()
 
# Create data
#np.random.seed(1)
#mu = [0,0]
#sigma = [[6,5], [5,6]]
# n = 1000
# x = np.random.multivariate_normal(mu, sigma, size=n)
file = open("data/HAR/test.csv")
x = np.loadtxt(file, delimiter=",", skiprows=1)
 
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
# Note: Use w.real.round(4) to (1) remove 'j' notation to real, (2) round to '4' significant digits
# np.savetxt('data/Result/Eigenvalues/Eigenvalues.txt', w, delimiter=" ", fmt="%s") 
# with open("data/Result/Eigenvalues/Eigenvectors.csv", 'w', newline='') as csvfile: 
#     csvwriter = csv.writer(csvfile) 
#     csvwriter.writerows(v)

# Calculate inverse square root of Eigenvalues
# Optional: Add '.1e5' to avoid division errors if needed
# Create a diagonal matrix
diagw = np.diag(1/((w+.1e-5)**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
diagw = diagw.real.round(4) #convert to real and round off
# with open("data/Result/Eigenvalues/diagonal_matrix.csv", 'w', newline='') as csvfile: 
#     csvwriter = csv.writer(csvfile) 
#     csvwriter.writerows(diagw)
            
# Calculate Rotation (optional)
# Note: To see how data can be rotated
xrot = np.dot(v, xc)
 
# Whitening transform using PCA (Principal Component Analysis)
wpca = np.dot(np.dot(diagw, v.T), xc)
wpca = wpca.real.round(4)
PCA_time = time.time() - start_time
 
# Whitening transform using ZCA (Zero Component Analysis)
wzca = np.dot(np.dot(np.dot(v, diagw), v.T), xc)
wzca = wzca.real.round(4)
ZCA_time = time.time() - start_time

# with open("data/Result/Eigenvalues/PCA_result.csv", 'w', newline='') as csvfile: 
#     csvwriter = csv.writer(csvfile) 
#     csvwriter.writerows(wpca)
# with open("data/Result/Eigenvalues/ZCA_result.csv", 'w', newline='') as csvfile: 
#     csvwriter = csv.writer(csvfile) 
#     csvwriter.writerows(wzca)
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- Time: %s seconds ---" % (time.time() - start_time))



