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
 
# Create data
json_data = []
json_file = "data/dim.json"
file = open(json_file)
for line in file:
 	json_line = json.loads(line)
 	json_data.append(json_line)
array = []
for i in range(len(json_data)):
    array.append(np.array(json_data[i]["target"]))
x = np.array(array)

# array = []
# for i in range(11):
#     app=[]
#     for j in range(10):
#         num = round(random.uniform(1,10), 4)
#         # num = randrange(10)
#         app.append(num)
#     array.append(np.array(app))
# x = np.array(array)

with open("data/Result/Eigenvalues/X.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(x)

# Zero center data
xc = x - np.mean(x, axis=0)
xc = xc.T

# Calculate Covariance matrix
# Note: 'rowvar=True' because each row is considered as a feature
# Note: 'bias=True' to divide the sum of squared variances by 'n' instead of 'n-1'
xcov = np.cov(xc, rowvar=True, bias=True)
with open("data/Result/Eigenvalues/Sigma.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(xcov.real.round(4))
 
# Calculate Eigenvalues and Eigenvectors
w, U = linalg.eig(xcov) # .eigh()
# w = w.real.round(4)
# v = v.real.round(4)
# Note: Use w.real.round(4) to (1) remove 'j' notation to real, (2) round to '4' significant digits
np.savetxt('data/Result/Eigenvalues/Eigenvalues.txt', w.real.round(4), delimiter=" ", fmt="%s") 
with open("data/Result/Eigenvalues/U.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(U.real.round(4))

Ut = U.T
with open("data/Result/Eigenvalues/U^T.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Ut.real.round(4))

# Calculate inverse square root of Eigenvalues
# Optional: Add '.1e5' to avoid division errors if needed
# Create a diagonal matrix
# Diagonal matrix for inverse square root of Eigenvalues
diagw = np.diag(1/((w+.1e-5)**0.5)) # np.diag(1/(w**0.5)) or np.diag(1/((w+.1e-5)**0.5))
#diagw = diagw.real.round(4) #convert to real and round off

d = ''
with open("data/Result/Eigenvalues/Λ^-1%2.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(diagw.real.round(4))
            
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
I_pca = np.dot(np.dot(W, Sig), Wt)
Sig_inv = np.dot(np.dot(U, diagw), Ut)
I_zca = np.dot(np.dot(Sig_inv, Sig), Sig_inv.T)

psp = np.dot(U.T, np.dot(Sig, U))
with open("data/Result/psp.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(psp.real.round(4))
    
Lambda = np.diag(w)
with open("data/Result/Lambda.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Lambda.real.round(4))

covarianza = np.cov(wpca)
with open("data/Result/covarianza.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(covarianza.real.round(0))

with open("data/Result/Eigenvalues/W.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(W.real.round(4))
with open("data/Result/Eigenvalues/W^T.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Wt.real.round(4))
with open("data/Result/Eigenvalues/W Sigma W^T.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(I_pca.real.round(4))
with open("data/Result/Eigenvalues/Sigma^-1%2.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Sig_inv.real.round(4))
with open("data/Result/Eigenvalues/Sigma^-1%2 Sigma Sigma^-1%2^T.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(I_zca.real.round(4))

cov_PCA = np.cov(wpca, rowvar=True, bias=True)
with open("data/Result/Eigenvalues/cov_PCA.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(cov_PCA.real.round(4))
cov_ZCA = np.cov(wzca, rowvar=True, bias=True)
with open("data/Result/Eigenvalues/cov_ZCA.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(cov_ZCA.real.round(4))
    
WSigPCA = np.dot(W, np.dot(np.dot(Sig, Wt), W))
WSigZCA = np.dot(Sig_inv, np.dot(np.dot(Sig, Sig_inv.T), Sig_inv))
with open("data/Result/Eigenvalues/W Sig Wt W PCA.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(WSigPCA.real.round(4))
with open("data/Result/Eigenvalues/W Sig Wt W ZCA.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(WSigZCA.real.round(4))
    
WtW = np.dot(Wt, W)
SigtSig = np.dot(Sig_inv.T, Sig_inv)
Sig_inverso = np.linalg.inv(Sig)
with open("data/Result/Eigenvalues/WtW.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(WtW.real.round(4))
with open("data/Result/Eigenvalues/SigmatSigma.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(SigtSig.real.round(4))
with open("data/Result/Eigenvalues/Sigma^-1.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(Sig_inverso.real.round(4))

with open("data/Result/Eigenvalues/PCA_result.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(wpca.real.round(4))
with open("data/Result/Eigenvalues/ZCA_result.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(wzca.real.round(4))
print("--- Columns: %d ---" % len(x[0]))
print("--- Rows: %d ---" % len(x))
print("--- Time: %s seconds ---" % (time.time() - start_time))








