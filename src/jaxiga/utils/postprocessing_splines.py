#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocessing spline subroutines

@author: cosmin
"""
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def plot_basis_or_deriv(x_plot, p2e_plot, xN_plot, dim_space, IEN, deg, plot_title=""):
    '''
    Plots the (spline) basis functions or their derivatives

    Parameters
    ----------
    x_plot : (1D array of floats)
        coordinates of the plot points in the parameter space
    p2e_plot : (array of integers)
        point to knot-span connectivity
    xN_plot : (array of size len(x_plot)x(deg+1))
        values of the (deg+1) basis functions or derivatives at each plot point
    dim_space : (int)
        dimension of the approximation space (number of spline basis functions)
    IEN : (2D array of ints)
        element to node connectivity
    deg : (int)
        polynomial degree of the basis
    plot_title : (string, optional)
        title of the plot

    Returns
    -------
    None.

    '''
    num_plot_pts = len(x_plot)
    plot_vals = jnp.zeros((num_plot_pts, 0))
    for i in range(dim_space):
        plot_col = jnp.zeros((0, 1))    
        for j in range(num_plot_pts):
            k = p2e_plot[j]
            if jnp.isin(i, IEN[k]):
                for l in range(deg+1):
                    if IEN[k,l] == i:
                        plot_col = jnp.vstack((plot_col, xN_plot[j,l]))
                        break
            else:
                plot_col = jnp.vstack((plot_col, 0))  
        plot_vals = jnp.concatenate((plot_vals, plot_col), axis=1)
    for i in range(dim_space):
        plt.plot(x_plot, plot_vals[:,i])
    plt.title(plot_title)
    plt.show()
    
def plot_solution_or_deriv(sol0, x_plot, p2e_plot, uhat_plot, xN_plot, IEN, deg,
                           plot_title=""):
    '''
    Plots the solution given the solution vector sol0

    Parameters
    ----------
    sol0 : (1D array of floats)
        the solution vector
    x_plot : (list of floats)
        evaluation points
    p2e_plot : (array of integers)
        point to knot-span connectivity
    uhat_plot : (1D array of floats)
        coordinates of the plot points in the reference coordinates (generated
                                                 by map_parameter_to_reference)
    xN_plot : (array of size len(x_plot)x(deg+1))
        values of the (deg+1) basis functions or derivatives at each plot point
    IEN : (2D array of ints)
        element to node connectivity
    deg : (int)
        polynomial degree of the basis    
    plot_title : (string, optional)
        title of the plot

    Returns
    -------
    None.

    '''
    num_plot_pts = len(x_plot)
    plot_vals = jnp.zeros((num_plot_pts))
    for i in range(num_plot_pts):
        k = p2e_plot[i]
        plot_val = 0
        for j in range(deg+1):
            plot_val += sol0[IEN[k,j]]*xN_plot[i, j]
        plot_vals = plot_vals.at[i].set(plot_val.item())
            
    plt.plot(x_plot, plot_vals)
    plt.title(plot_title)
    plt.show()

def plot_error_solution_or_deriv(sol0, x_plot, p2e_plot, uhat_plot, xN_plot, IEN,
                                 deg, exact_sol_fun, plot_title=""):
    '''
    Plots the error for the solution given the solution vector sol0

    Parameters
    ----------
    sol0 : (1D array of floats)
        the solution vector
    x_plot : (list of floats)
        evaluation points
    p2e_plot : (array of integers)
        point to knot-span connectivity
    uhat_plot : (1D array of floats)
        coordinates of the plot points in the reference coordinates (generated
                                                 by map_parameter_to_reference)
    xN_plot : (array of size len(x_plot)x(deg+1))
        values of the (deg+1) basis functions or derivatives at each plot point    
    IEN : (2D array of ints)
        element to node connectivity
    deg : (int)
        polynomial degree of the basis    
    exact_sol_fun : (function)
            function name for the exact solution or derivative thereof
    plot_title : (string, optional)
        title of the plot

    Returns
    -------
    None.

    '''
    num_plot_pts = len(x_plot)
    comp_sol_vals = jnp.zeros((num_plot_pts))
    exact_sol_vals = exact_sol_fun(x_plot)
    for i in range(num_plot_pts):
        k = p2e_plot[i]
        comp_sol_val = 0
        for j in range(deg+1):
            comp_sol_val += sol0[IEN[k,j]]*xN_plot[i, j]
        comp_sol_vals = comp_sol_vals.at[i].set(comp_sol_val.item())
    plot_vals = exact_sol_vals - comp_sol_vals        
     
    # plt.plot(x_plot, plot_col)
    # plt.show()
    #input("Press Enter to continue...")
    plt.plot(x_plot, plot_vals)
    plt.title(plot_title)
    plt.show()
    

