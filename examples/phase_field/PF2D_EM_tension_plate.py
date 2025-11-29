#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM with splines basis 

@author: cosmin
"""


import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import config
import time
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow_probability as tfp

from jaxiga.utils.Geom_examples import Quadrilateral
from jaxiga.utils_iga.materials import MaterialElast2D
from jaxiga.utils.IGA import IGAMesh2D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming
from jaxiga.utils.Solvers import PF2D_DEM_Spline
from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils.lbfgs import minimize as lbfgs_minimize
from jaxiga.utils.boundary import boundary2D, get_bcdof_bcval
from jaxiga.utils.processing_splines import evaluate_spline_basis_fem_2d
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils_iga.postprocessing import (plot_sol2D_elast,
                                      plot_sol2D,
                                      comp_measurement_values,
                                      get_measurements_vector,
                                      plot_fields_2D)
from jaxiga.utils.phase_field import history_edge_crack
from jaxiga.utils.postprocessing import plot_scattered_tricontourf


key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 1
num_refinements = 6
output_filename = "pf2d_tension_plate_DEM"


# Step 0: Define the material and model parameters
Emod = 210e3 # Young's modulus
nu = 0.3 # Poisson ratio
B = 1e3 # Parameter for initial history function
l = 0.0125*2 # Length parameter which controls the spread of the damage
cenerg = 2.7 # Critical energy release for unstable crack (Gc)
material = MaterialElast2D(Emod=Emod, nu=nu, plane_type="stress")

num_load_steps = 10
delta_disp = 1e-3


# symmetry Dirichlet B.C., u_y = 0 for y=0 and u_x=0 for x=0
def u_bound_dir_fixed(x, y):
    return [0.0, 0.0]

def u_bound_dir_vert_disp(x, y, disp=0.):
    return [0.0, disp]


# # Set the boundary conditions
bound_down = boundary2D("Dirichlet", 0, "down", u_bound_dir_fixed)
bound_up = boundary2D("Dirichlet", 0, "up", u_bound_dir_vert_disp)
bound_cond = [bound_down, bound_up]

# Step 1: Generate the geometry
# Patch 1:
_, ax = plt.subplots()
plate_length = 1.
plate_width  = 1.
vertices = [[0., 0.], [0., plate_width], [plate_length, 0.], [plate_length, plate_width]]
patch1 = Quadrilateral(np.array(vertices))
patch1.plotKntSurf(ax)

# Step 2: Degree elevate and refine the geometry
t = time.time()
patch1.degreeElev(deg - 1, deg - 1)
elapsed = time.time() - t
print("Degree elevation took ", elapsed, " seconds")

t = time.time()
for i in range(num_refinements):
    patch1.refine_knotvectors(True, True)

elapsed = time.time() - t
print("Knot insertion took ", elapsed, " seconds")

_, ax = plt.subplots()
patch1.plotKntSurf(ax)
plt.show()

t = time.time()
mesh1 = IGAMesh2D(patch1)
elapsed = time.time() - t
print("Mesh initialization took ", elapsed, " seconds")

mesh1.classify_boundary()
meshes = [mesh1]
vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(meshes)
edge_list = gen_edge_list(patch2vertex)
size_basis = zip_conforming(meshes, vertex2patch, edge_list)

# Generate the Gauss points
# Evaluate the spline basis
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_quad_v = gen_gauss_pts(deg + 1)
gauss_rule = [gauss_quad_u, gauss_quad_v]
num_fields = 2
R, dR, local_areas, phys_pts, global_nodes_u = evaluate_spline_basis_fem_2d(meshes,
                                                                              gauss_rule,
                                                                              num_fields)

# History function and phase field initialization.
num_fields = 1
_, _, _, phys_pts, global_nodes_phi = evaluate_spline_basis_fem_2d(meshes,
                                                    gauss_rule,
                                                    num_fields)
fenerg = history_edge_crack(phys_pts, B, l, cenerg)
local_areas = jnp.reshape(local_areas, (-1, 1))

# # prepare a mask which sets the phase fields energy to zero for y>y_thres
# y_thres = 0.9
# mask = jnp.where(phys_pts[:,:,1]>y_thres, 0., 1.)
mask = jnp.ones_like(phys_pts[:,:,1])

bcdof, _ = get_bcdof_bcval(meshes, bound_cond)
model = PF2D_DEM_Spline(global_nodes_u, global_nodes_phi, deg, size_basis, bcdof,
                           material, l, cenerg, mask)

top_disp = 0.
for i_load in range(num_load_steps):
    
    # Get the boundary degrees of freedom and set them to zeros
    bcdof, bcval = get_bcdof_bcval(meshes, bound_cond)
    bcval = jnp.expand_dims(bcval, axis=-1)
    
    
    phys_pts_all = jnp.reshape(phys_pts, (-1, 2))
    
    num_basis = (model.deg+1)**2
    
    model.train(R, dR, jnp.squeeze(local_areas), bcval, fenerg, n_iter=5000)
    
    t_start = time.time()
    params = model.get_params(model.opt_state)
    loss_func = jax_tfp_function_factory(model,  params, R, dR,
                                         jnp.squeeze(local_areas), bcval, fenerg)
    initial_pos = loss_func.init_params_1d
    current_loss, _ = loss_func(initial_pos)
    print("Initial loss is ", current_loss)
    num_bfgs_iterations = 0
    
    results = lbfgs_minimize(loss_func, initial_position=initial_pos,
                              num_correction_pairs=100, max_iterations=10000)
    initial_pos = results.position
    num_bfgs_iterations += results.num_iterations
    print("Iteration: ", num_bfgs_iterations, " loss: ", results.objective_value)
    
    print("Time taken (BFGS) is ", time.time() - t_start)
    
    
    
    _, unflatten_params = jax.flatten_util.ravel_pytree(params)
    params = unflatten_params(results.position)
    
    file_name = 'params_step_'+str(i_load)+'.npy'
    jnp.save(file_name, params)
    print('Parameters saved in', file_name)
    
    energ_int, energy_phi_1, energy_phi_2, fenerg, phi_dens, energ_dens = model.get_all_losses(params, R, dR,
                                                                 jnp.squeeze(local_areas),
                                                                 bcval, fenerg)
    print("Energy_int =", energ_int)
    print("Energy phi 1 =", energy_phi_1)
    print("Energy phi 2 =", energy_phi_2)
    
    plot_scattered_tricontourf(phys_pts_all[:,0], phys_pts_all[:,1], phi_dens.flatten())
    plt.title('Phi density at step ' + str(i_load))
    plt.show()
    
    plot_scattered_tricontourf(phys_pts_all[:,0], phys_pts_all[:,1], energ_dens.flatten())
    plt.title('Energy density at step '+ str(i_load))
    plt.show()
    
    num_free_u = len(model.index_map) - len(bcdof)
    interior_vals_u = unflatten_params(results.position)[:num_free_u]
    interior_vals_phi = unflatten_params(results.position)[num_free_u:]
    
    sol_u = jnp.zeros((2*size_basis))
    sol_u = sol_u.at[bcdof].set(bcval.flatten())
    sol_u = sol_u.at[model.trainable_indx].set(interior_vals_u[:, 0])
    sol_u = np.array(sol_u)
    
    sol_phi = np.array(interior_vals_phi)
    
    t = time.time()
    plot_sol2D_elast(meshes, material.Cmat, sol_u, output_filename+'_u_step_'+str(i_load))
    plot_sol2D(meshes, sol_phi, output_filename+'_phi_step_'+str(i_load))
    elapsed = time.time() - t
    print("Plotting to VTK took ", elapsed, " seconds")
    
    # compute the solution at a set of uniformly spaced points
    num_pts_xi = 100
    num_pts_eta = 100
    num_fields = 1
    meas_phi_all, meas_pts_phys_xy_all, vals_phi_min, vals_phi_max = comp_measurement_values(num_pts_xi,
                                                                  num_pts_eta,
                                                                  meshes,
                                                                  sol_phi,
                                                                  get_measurements_vector,
                                                                  num_fields)
    
    num_fields = 2
    meas_disp_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = comp_measurement_values(num_pts_xi,
                                                                  num_pts_eta,
                                                                  meshes,
                                                                  sol_u,
                                                                  get_measurements_vector,
                                                                  num_fields)
    elapsed = time.time() - t
    print("Computing the values at measurement points took ", elapsed, " seconds")
    
    t = time.time()
    field_title = "Computed solution"
    
    field_names = ['phi']
    plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes, 
                    meas_pts_phys_xy_all, meas_phi_all, vals_phi_min, vals_phi_max)
    
    
    field_names = ['x-disp', 'y-disp']
    plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes, 
                    meas_pts_phys_xy_all, meas_disp_all, vals_disp_min, vals_disp_max)
    elapsed = time.time() - t
    print("Plotting the solution (matplotlib) took ", elapsed, " seconds")
    
    # Update the boudnary conditions
    top_disp += delta_disp
    u_bound_dir_var_vert_disp = lambda x, y : [0.0, top_disp]
    # # Set the boundary conditions    
    bound_up = boundary2D("Dirichlet", 0, "up", u_bound_dir_var_vert_disp)
    bound_cond = [bound_down, bound_up]
