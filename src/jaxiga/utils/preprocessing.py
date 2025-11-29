#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 22:59:13 2022

@author: cosmin
"""

import jax.numpy as jnp
from functools import partial
from jax import random, vmap, jit
from jax.config import config
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from jax.experimental.ode import odeint



# Length scale of a Gaussian random field (GRF)
length_scale = 0.1

def AntiderivativeSolver(nodes, f, with_endpoints = False):
    '''
    Solves the antiderivative problem:
        -u'(x) = f(x) for x_min < x <= x_max    
    with Dirichlet boundary conditions
    u(x_min) = 0

    Parameters
    ----------
    nodes : (1D array of length N+1)
        location of the FEM nodes with 
            x_min = nodes[0] < nodes[1] < ... < nodes[N-1] < nodes[N] = x_max
    f :  (1D array of length N)
        values of f(x) at x=nodes[1], ..., nodes[N]
    
    Returns
    -------
    u : (1D array of length N-1)
        values of u(x) at  x=nodes[1], ..., nodes[N-1]

    '''
    
    dim_space = len(nodes) - 1
    rhs = jnp.zeros(dim_space)
    rhs = rhs.at[:-1].set((nodes[2:dim_space+1]-nodes[0:dim_space-1])/2*f[:-1])
    rhs = rhs.at[-1].set((nodes[-1]-nodes[-2])/2*f[-1])
    # rhs[:-1] = (nodes[2:dim_space+1]-nodes[0:dim_space-1])/2*f[:-1]
    # rhs[-1] = (nodes[-1]-nodes[-2])/2*f[-1]
    lhs_off_diag_up = 1/2*jnp.ones(dim_space-1)
    lhs_off_diag_down = -1/2*jnp.ones(dim_space-1)
    lhs_diag = jnp.zeros(dim_space)
    lhs_diag = lhs_diag.at[-1].set(0.5)
    
    lhs = diags([lhs_diag, lhs_off_diag_down, lhs_off_diag_up], [0,-1,1], format='csr')
    u = spsolve(lhs, rhs)
    
    return u

def Poisson1DSolver(nodes, f, with_endpoints = False, left_bnd = 0., right_bnd = 0.):
    '''
    Solves the 1D Poisson problem:
        -u''(x) = f(x) for x_min < x <= x_max    
    with Dirichlet boundary conditions
    u(x_min) = left_bnd, u(x_max) =  right_bnd

    Parameters
    ----------
    nodes : (1D array of length N+1)
        location of the FEM nodes with 
            x_min = nodes[0] < nodes[1] < ... < nodes[N-1] < nodes[N] = x_max
    f :  (1D array of length N-1)
        values of f(x) at x=nodes[1], ..., nodes[N-1]
        
    with_endpoints : (boolean)
        if True then the input set also includes f(x_min) and f(x_max)
        
    left_bnd : (float)
            the Dirichlet boundary condition, u(x_min) = left_bnd
        
    right_bnd : (float)
            the Dirichlet boundary condition u(x_max) = right_bnd
    
    Returns
    -------
    u : (1D array of length N-1)
        values of u(x) at  x=nodes[1], ..., nodes[N-1]

    '''
    if with_endpoints:
        f = f[1:-1]
    
    dim_space = len(nodes) - 2    
    left_bnd_arr = jnp.array([left_bnd])
    right_bnd_arr = jnp.array([right_bnd])
    zero_arr = jnp.array([0.])
    one_arr = jnp.array([1.])
    rhs = jnp.concatenate([left_bnd_arr, (nodes[2:dim_space+2]-nodes[0:dim_space])/2*f,
                           right_bnd_arr])
    lhs_off_diag_top = jnp.concatenate([zero_arr, -1/(nodes[2:dim_space+2]-nodes[1:dim_space+1])])
    lhs_off_diag_bottom = jnp.concatenate([-1/(nodes[1:dim_space+1]-nodes[0:dim_space]), zero_arr])
    lhs_diag_left = 1/(nodes[1:dim_space+1]-nodes[0:dim_space])
    lhs_diag_right =  1/(nodes[2:dim_space+2]-nodes[1:dim_space+1])
    lhs_diag = jnp.concatenate([one_arr, lhs_diag_left + lhs_diag_right, one_arr])
    lhs = diags([lhs_diag, lhs_off_diag_bottom, lhs_off_diag_top], [0,-1,1], format='csr')
   
    u = spsolve(lhs, rhs)    
    
    return u[1:-1]


# Define RBF kernel
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)

# Geneate training data corresponding to one input sample
def generate_one_input_output(key, nodes, interp_nodes, solver, length_scale = 0.1,
                              with_endpoints=False, left_bnd = 0., right_bnd = 0.):
    # Sample GP prior at a fine grid
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, 1, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(N))
    gp_sample = jnp.dot(L, random.normal(key, (N,)))

    # Input sensor locations and measurements
    f_train = jnp.interp(interp_nodes, X.flatten(), gp_sample)

    # Output sensor locations and measurements
    u_train = solver(nodes, f_train, with_endpoints = with_endpoints, 
                     left_bnd = left_bnd, right_bnd = right_bnd)

    
    return f_train, u_train


# Geneate training data corresponding to N input sample
def generate_training_data(key, N, nodes, interp_nodes, dim_space, solver,
                           with_endpoints=False, left_bnd = 0., 
                           right_bnd = 0.):
    print("Generating data set of size ", N)

    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    #gen_fn = lambda key: generate_one_input_output(key, nodes)
    #f_train = vmap(gen_fn)(keys)
    
    
    f_train_all = jnp.zeros((N, len(interp_nodes)))
    u_train_all = jnp.zeros((N, dim_space))
    for i in range(N):
        f_train, u_train = generate_one_input_output(keys[i], nodes, interp_nodes,
                                        solver, with_endpoints = with_endpoints, 
                                        left_bnd = left_bnd, right_bnd = right_bnd)
        u_train_all = u_train_all.at[i, :].set(u_train)
        f_train_all = f_train_all.at[i, :].set(f_train)

    config.update("jax_enable_x64", False)
    return u_train_all, f_train_all

def generate_piecewise_linear_data(nodes, interp_nodes, solver, left_bnd = 0.,
                                   right_bnd = 0.):
    config.update("jax_enable_x64", True)
    dim_space = len(interp_nodes)
    print("Generating data set of size ", len(interp_nodes))

    f_train_all = jnp.zeros((dim_space, dim_space))
    u_train_all = jnp.zeros((dim_space, dim_space))
    
    for i in range(dim_space):
        f_train = jnp.zeros(dim_space)
        f_train = f_train.at[i].set(1.)
        u_train = solver(nodes, f_train, left_bnd = left_bnd, right_bnd = right_bnd)
        u_train_all = u_train_all.at[i, :].set(u_train)
        f_train_all = f_train_all.at[i, :].set(f_train)
    config.update("jax_enable_x64", False)
    return u_train_all, f_train_all

