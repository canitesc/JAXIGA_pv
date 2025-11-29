#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for plotting, error norm computations and other post-processing tasks
in one dimension
"""
import numpy as np
from jaxiga.utils.bernstein import bernstein_basis_1d


def comp_measurement_values_1d(num_pts_xi, mesh_list, sol0, meas_func,
                            num_fields, *params):
    meas_pts_phys_x_all = []
    meas_vals_all = []
    vals_min = []
    vals_max = []
    for _ in range(num_fields):
        meas_vals_all.append([])
        vals_min.append(float('inf'))
        vals_max.append(float('-inf'))
    for i in range(len(mesh_list)):    
        meas_points_param_xi = np.linspace(0, 1, num_pts_xi)        
        
        meas_pts_param_xi_i = np.zeros((len(meas_points_param_xi),2))
        row_counter = 0
        for pt_xi in meas_points_param_xi:            
                #meas_pts_param_xi_eta_i.at[row_counter, :].set([pt_xi, pt_eta, i])
                meas_pts_param_xi_i[row_counter, :] = [pt_xi, i]
                row_counter += 1
        
        meas_pts_phys_x, meas_vals = meas_func(mesh_list, sol0, 
                                                meas_pts_param_xi_i, num_fields, *params)    
        meas_pts_phys_x_all.append(meas_pts_phys_x)
        for i_field in range(num_fields):
            meas_vals_all[i_field].append(meas_vals[i_field])
            vals_min[i_field] = np.minimum(vals_min[i_field], np.min(meas_vals[i_field]))
            vals_max[i_field] = np.maximum(vals_max[i_field], np.max(meas_vals[i_field]))
    return meas_vals_all, meas_pts_phys_x_all, vals_min, vals_max


def get_measurements_vector_1d(mesh_list, sol, meas_pts_param_xi_i, num_fields):
    """
    Generates values of measurements from a given mesh and solution and a 
    given list measurement points in parameter space for a multi-field solution
    It is assumed that the sol contains a vector of the form 
    [u_0, v_0, ..., u_1, v_1, ...]

    Parameters
    ----------
    mesh_list : (list of IGAMesh1D) multipatch mesh
    sol : 1D array
        solution vector. 
    meas_pts_param_xi_i : (2D array)
        measurements points in the parameter space with one (u) coordinate
        and patch index in each row
    num_fields : (int) number of fields in the solution 

    Returns
    -------
    meas_pts_phys_x : (2D array)
        measurements points in the physical space with one (x) coordinate 
        in each row
    meas_val : (list of 1D array)
        the values of the solution computed at each measurement point

    """
    num_pts = len(meas_pts_param_xi_i)
    meas_vals = []
    for _ in range(num_fields):
        meas_vals.append(np.zeros(num_pts))    
    meas_pts_phys_x = np.zeros((num_pts, 1))
    for i_pt in range(num_pts):
        pt_xi_i = meas_pts_param_xi_i[i_pt]
        xi_coord = pt_xi_i[0]
        patch_index = int(pt_xi_i[1])
        for i in range(len(mesh_list[patch_index].elem_vertex)):
            elem_vertex = mesh_list[patch_index].elem_vertex[i]
            xi_min = elem_vertex[0]
            xi_max = elem_vertex[1]            
            if xi_min <= xi_coord and xi_coord <= xi_max:
                
                # map point to the reference element (i.e. mapping from 
                # (eta_min, eta_max) and (xi_min, v=xi_max) to (-1, 1)
                local_nodes = mesh_list[patch_index].elem_node[i]
                global_nodes = mesh_list[patch_index].elem_node_global[i]
                cpts = mesh_list[patch_index].cpts[0:1, local_nodes]
                wgts = mesh_list[patch_index].wgts[local_nodes]
                u_coord = 2/(xi_max-xi_min)*(xi_coord-xi_min) - 1                
                Bu, _ = bernstein_basis_1d(np.array([u_coord]), 
                                           mesh_list[patch_index].deg)
                
                # compute the (B-)spline basis functions and derivatives with
                # Bezier extraction
                N_mat = mesh_list[patch_index].C[i] @ Bu[0, :]
                RR = N_mat * wgts
                w_sum = np.sum(RR)
                RR /= w_sum
                meas_pts_phys_x[i_pt,:] = cpts @ RR
                for i_field in range(num_fields):
                    meas_vals[i_field][i_pt] = np.dot(RR,
                                                      sol[num_fields*global_nodes+i_field])
                break    
    return meas_pts_phys_x, meas_vals