#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:32:31 2024

@author: cosmin
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
l0 = 0.25
x_start, x_end = -2, 2
n_points = 101  # Number of points in the grid
dx = (x_end - x_start) / (n_points - 1)  # Distance between adjacent grid points

# Grid
x = np.linspace(x_start, x_end, n_points)

# Construct the coefficient matrix A for the discretized version of l0^2 * u''(x) - u(x) = 0
# Using the finite difference method for the second derivative approximation
diagonal = -2*l0**2/dx**2 - 1
off_diagonal = l0**2/dx**2
A = diags([off_diagonal, diagonal, off_diagonal], [-1, 0, 1], shape=(n_points, n_points)).toarray()

# Find the index closest to x=0
index_at_zero = np.argmin(np.abs(x))

# Resetting b to handle the condition u(0) = 1 more explicitly
b = np.zeros(n_points)
b[index_at_zero] = 1  # Directly setting the condition u(0) = 1

# Adjust A to directly incorporate this condition
A = diags([off_diagonal, diagonal, off_diagonal], [-1, 0, 1], shape=(n_points, n_points)).toarray()
A[index_at_zero, :] = 0
A[index_at_zero, index_at_zero] = 1

# Solve the linear system again with the adjusted setup
u_corrected = spsolve(A, b)

# Plot the corrected solution
plt.figure(figsize=(10, 6))
plt.plot(x, u_corrected, label='Corrected Numerical Solution')
plt.scatter(x[index_at_zero], u_corrected[index_at_zero], color='red', label='u(0) = 1 Condition')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Corrected Solution with u(0) = 1')
plt.legend()
plt.grid(True)
plt.show()