#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics informed neural network based on colocation with splines basis

@author: cosmin
"""
#%%
#%config InlineBackend.figure_format = "retina"

import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import config
import time
import matplotlib.pyplot as plt
import numpy as np

from jaxiga.utils.Geom_examples import Quadrilateral
from jaxiga.utils.IGA import IGAMesh2D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming
from jaxiga.utils.Solvers import Poisson2D_PINN_Spline
from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils.boundary import boundary2D
from jaxiga.utils.bfgs import minimize as bfgs_minimize
from jaxiga.utils.preprocessing_splines import (generate_Greville_abscissae2D, 
                                         make_point_to_element_connectivity_2d,
                                         get_bcdof,
                                         get_boundary_indices,
                                         get_interior_indices)
from jaxiga.utils.processing_splines import evaluate_spline_basis_2d
from jaxiga.utils_iga.postprocessing import (plot_sol2D,
                                      comp_measurement_values,
                                      get_measurements_vector,
                                      plot_fields_2D,
                                      comp_error_2D,
                                      comp_error_norm)
key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 5
num_refinements = 5

# Step 0: Define the function and boundary conditions
def f(x, y):
    return 8 * jnp.pi ** 2 * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)

def u_bound(x, y):
    return 0.0

def a0(x, y):
    return 1.0

# The exact solution for error norm computations
def exact_sol(x, y):
    return (jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y),)

def deriv_exact_sol(x, y):
    return [
        2 * jnp.pi * jnp.cos(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y),
        2 * jnp.pi * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y),
    ]

bound_left = boundary2D("Dirichlet", 0, "left", u_bound)
bound_right = boundary2D("Dirichlet", 0, "right", u_bound)
bound_down = boundary2D("Dirichlet", 0, "down", u_bound)
bound_up = boundary2D("Dirichlet", 0, "up", u_bound)
bound_cond = [bound_left, bound_right, bound_down, bound_up]


# Generate the geometry
# Patch 1:
corners = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
patch1 = Quadrilateral(corners)

# Step 2: Degree elevate and refine the geometry
t = time.time()
patch1.degreeElev(deg - 1, deg - 1)
elapsed = time.time() - t
print("Degree elevation took ", elapsed, " seconds")

t = time.time()
for i in range(num_refinements):
    patch1.refine_knotvectors(True, True)
#patch1.refine_knotvectors(True, False)
elapsed = time.time() - t
print("Knot insertion took ", elapsed, " seconds")

_, ax = plt.subplots()
patch1.plotKntSurf(ax)

t = time.time()
mesh1 = IGAMesh2D(patch1)
elapsed = time.time() - t
print("Mesh initialization took ", elapsed, " seconds")
# plt.scatter(mesh1.cpts[0,:], mesh1.cpts[1,:])

mesh1.classify_boundary()
mesh_list = [mesh1]
vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(mesh_list)
edge_list = gen_edge_list(patch2vertex)
size_basis = zip_conforming(mesh_list, vertex2patch, edge_list)
patch_list = [patch1]

# Generate collocation points (training data for PINN)
# Generate Greville abscissae
t = time.time()
generate_Greville_abscissae2D(patch_list)
elapsed = time.time() - t
print("Generating Grevvile abscissae took ", elapsed, " seconds")
plt.show()

t = time.time()
make_point_to_element_connectivity_2d(patch_list)
        
# Evaluate the spline basis
evaluate_spline_basis_2d(patch_list, mesh_list)

# Get the boundary degrees of freedom and set them to zeros
bcdof = get_bcdof(bound_cond, mesh_list)
bcval = jnp.zeros(len(bcdof))
# Get the indices for the boundary points and interior points
# TODO: Fix for multipatch
pt_bound_indices = get_boundary_indices(patch1.num_pts_u, patch1.num_pts_v)
pt_interior_indices, interior_indices = get_interior_indices(bcdof, 
                                                             patch1.num_pts_uv,
                                                             pt_bound_indices)

fvals = f(patch1.G_uv[pt_interior_indices, 0], 
          patch1.G_uv[pt_interior_indices, 1])[:, jnp.newaxis]
IEN = mesh1.elem_node_global
model = Poisson2D_PINN_Spline(IEN, patch1.p2e_uv[pt_interior_indices], deg,
                              size_basis, bcdof, bcval)
ddMx = mesh1.ddMu[pt_interior_indices]
ddMy = mesh1.ddMv[pt_interior_indices]

elapsed = time.time() - t
print("Computing derivatives took ", elapsed, " seconds")

model.train(ddMx, ddMy, fvals, n_iter = 100)

t_start = time.time()
params = model.get_params(model.opt_state)
loss_func = jax_tfp_function_factory(model,  params, ddMx, ddMy, fvals)
initial_pos = loss_func.init_params_1d
tolerance = 1e-5
current_loss, _ = loss_func(initial_pos)
print("Initial loss is ", current_loss)
num_bfgs_iterations = 0
while current_loss > tolerance:
    results = bfgs_minimize(loss_func, initial_position = initial_pos,
                        max_iterations=1000)
    initial_pos = results.position
    num_bfgs_iterations += results.num_iterations
    print("Iteration: ", num_bfgs_iterations, " loss: ", results.objective_value)
    if current_loss < results.objective_value-tolerance:
        current_loss = results.objective_value
    else:
        break
    
print("Time taken (BFGS) is ", time.time() - t_start)

_, unflatten_params = jax.flatten_util.ravel_pytree(params)
interior_vals = unflatten_params(results.position)

sol0 = jnp.zeros((size_basis))
sol0 = sol0.at[bcdof].set(bcval)
sol0 = sol0.at[interior_indices].set(interior_vals[:,0])

t = time.time()
output_filename = "poisson_2d_square"
plot_sol2D(mesh_list, sol0, output_filename)
elapsed = time.time() - t
print("Plotting to VTK took ", elapsed, " seconds")

t = time.time()
# compute the solution at a set of uniformly spaced points
num_pts_xi = 100
num_pts_eta = 100
num_fields = 1
meas_vals_all, meas_pts_phys_xy_all, vals_min, vals_max = comp_measurement_values(num_pts_xi,
                                                              num_pts_eta,
                                                              mesh_list,
                                                              sol0,
                                                              get_measurements_vector,
                                                              num_fields)
elapsed = time.time() - t
print("Computing the values at measurement points took ", elapsed, " seconds")

t = time.time()
field_title = "Computed solution"
field_names = ['U']
plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list, 
                meas_pts_phys_xy_all, meas_vals_all, vals_min, vals_max)
elapsed = time.time() - t
print("Plotting the solution (matplotlib) took ", elapsed, " seconds")

# compute the error at a set of uniformy spaced points
t = time.time()
err_vals_all, err_vals_min, err_vals_max = comp_error_2D(num_fields, 
                                                          mesh_list,
                                                          exact_sol, 
                                                          meas_pts_phys_xy_all,
                                                          meas_vals_all)
elapsed = time.time() - t
print("Computing the error at measurement points took ", elapsed, " seconds")

# plot the error as a contour plot
t = time.time()
field_title = "Error in the solution"
plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list, 
                meas_pts_phys_xy_all, err_vals_all, err_vals_min, err_vals_max)
elapsed = time.time() - t
print("Plotting the error (matplotlib) took ", elapsed, " seconds")


# Compute the norm of the error
t = time.time()
# Generate the Gauss points        
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_quad_v = gen_gauss_pts(deg + 1)
gauss_rule = [gauss_quad_u, gauss_quad_v]
rel_L2_err, rel_H1_err = comp_error_norm(mesh_list, sol0, exact_sol,
                                         deriv_exact_sol, a0,
                                         gauss_rule
)
print("Relative L2-norm error is ", rel_L2_err)
print("Relative energy-norm error is ", rel_H1_err)
elapsed = time.time() - t
print("Computing the error norms took", elapsed, " seconds")