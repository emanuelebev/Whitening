# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:16:27 2021

@author: emanu
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
 
# Create data
np.random.seed(1)
mu = [0,0]
sigma = [[6,5], [5,6]]
n = 1000
x = np.random.multivariate_normal(mu, sigma, size=n)
print('x.shape:', x.shape, '\n')
 
# Zero center data
xc = x - np.mean(x, axis=0)
print(xc.shape)
xc = xc.T
print('xc.shape:', xc.shape, '\n')
 
# Calculate Covariance matrix
# Note: 'rowvar=True' because each row is considered as a feature
# Note: 'bias=True' to divide the sum of squared variances by 'n' instead of 'n-1'
xcov = np.cov(xc, rowvar=True, bias=True)
print
 
# Calculate Eigenvalues and Eigenvectors
w, v = linalg.eig(xcov) # .eigh()
# Note: Use w.real.round(4) to (1) remove 'j' notation to real, (2) round to '4' significant digits
print("Eigenvalues:\n", w.real.round(4), '\n')
print("Eigenvectors:\n", v, '\n')
 
# Calculate inverse square root of Eigenvalues
# Optional: Add '.1e5' to avoid division errors if needed
# Create a diagonal matrix
diagw = np.diag(1/(w**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
diagw = diagw.real.round(4) #convert to real and round off
print("Diagonal matrix for inverse square root of Eigenvalues:\n", diagw, '\n')
 
# Calculate Rotation (optional)
# Note: To see how data can be rotated
xrot = np.dot(v, xc)
 
# Whitening transform using PCA (Principal Component Analysis)
wpca = np.dot(np.dot(diagw, v.T), xc)
 
# Whitening transform using ZCA (Zero Component Analysis)
wzca = np.dot(np.dot(np.dot(v, diagw), v.T), xc)
 
 
fig = plt.figure(figsize=(10,3))
 
plt.subplot(1,4,1)
plt.scatter(x[:,0], x[:,1], label='original', alpha=0.15)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
 
plt.subplot(1,4,2)
plt.scatter(xc[0,:], xc[1,:], label='centered', alpha=0.15)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
 
plt.subplot(1,4,3)
plt.scatter(wpca[0,:], wpca[1,:], label='wpca', alpha=0.15)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
 
plt.subplot(1,4,4)
plt.scatter(wzca[0,:], wzca[1,:], label='wzca', alpha=0.15)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
 
plt.tight_layout()
plt.savefig('whiten_2.png', dpi=300)