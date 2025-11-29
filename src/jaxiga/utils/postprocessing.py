#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:43:56 2024

@author: cosmin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def plot_convergence_dem(adam_loss_hist, bfgs_loss_hist, percentile=99., 
                         folder = None, file = None):
    '''
    Plots the convergence of the loss function for the Deep Energy Method. The
    values higher than the "percentile" parameter are ignored.

    Parameters
    ----------
    adam_loss_hist : (list of floats)
        the loss at each iteration of the Adam optimizer
    bfgs_loss_hist : (list of floats)
        the loss at each iteration of the BFGS optimizer
    percentile : (float)
        . The default is 99.

    Returns
    -------
    None.

    '''
    num_epoch = len(adam_loss_hist)
    num_iter_bfgs = len(bfgs_loss_hist)
    loss_hist_all = adam_loss_hist + bfgs_loss_hist
    y_max = np.percentile(loss_hist_all, percentile)
    y_min = np.min(loss_hist_all)
    plt.ylim(y_min, y_max)
    plt.plot(range(num_epoch), np.minimum(adam_loss_hist, y_max) , label='Adam')
    plt.plot(range(num_epoch, num_epoch+num_iter_bfgs), np.minimum(bfgs_loss_hist, y_max), 
                 label = 'BFGS')
    plt.legend()
    plt.title('Loss convergence')
    if folder != None:
        full_name = folder + '/' + file
        plt.savefig(full_name)
    plt.show()

def plot_scattered_tricontourf(x, y, z, cmap='viridis', levels=10, 
                              alpha=1.0, show_points=False, point_size=20, 
                              point_color='k', point_alpha=0.5,
                              colorbar=True, title=None, figsize=(10, 8)):
    """
    Create a filled contour plot from scattered data points using triangulation.
    
    Parameters:
    -----------
    x : array-like
        x-coordinates of the scattered data points
    y : array-like
        y-coordinates of the scattered data points
    z : array-like
        values at the scattered data points
    cmap : str or Colormap, optional
        colormap to use (default: 'viridis')
    levels : int or array-like, optional
        Number of contour levels or specific levels to plot (default: 10)
    alpha : float, optional
        transparency of the contour fill (default: 1.0)
    show_points : bool, optional
        whether to plot the original data points (default: False)
    point_size : float, optional
        size of the data points (default: 20)
    point_color : str or array-like, optional
        color of the data points (default: 'k')
    point_alpha : float, optional
        transparency of the data points (default: 0.5)
    colorbar : bool, optional
        whether to add a colorbar (default: True)
    title : str, optional
        title of the plot (default: None)
    figsize : tuple, optional
        figure size (width, height) in inches (default: (10, 8))
    
    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    ax : matplotlib Axes
        The created axes
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # Check if inputs have the same length
    if len(x) != len(y) or len(x) != len(z):
        raise ValueError("Input arrays x, y, and z must have the same length")
    
    # Create a Triangulation
    triang = Triangulation(x, y)
    
    # Create the figure and axis (2D)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the filled contour using triangulation
    contour = ax.tricontourf(triang, z, levels=levels, cmap=cmap, alpha=alpha)
    
    # Add the original data points if requested
    if show_points:
        scatter = ax.scatter(x, y, c=point_color, s=point_size, 
                            alpha=point_alpha, edgecolors='w')
    
    # Add a colorbar if requested
    if colorbar:
        fig.colorbar(contour, ax=ax)
    
    # Add a title if provided
    if title:
        ax.set_title(title)
    
    # Set equal aspect ratio for the plot
    ax.set_aspect('equal')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    return fig, ax