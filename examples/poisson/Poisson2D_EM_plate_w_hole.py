#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM with splines basis (i.e. IGA)

@author: cosmin
"""


import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import config
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
#import tensorflow_probability as tfp

from jaxiga.utils.Geom_examples import PlateWHoleQuadrant
from jaxiga.utils.preprocessing_splines import get_bcdof
from jaxiga.utils.IGA import IGAMesh2D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming
from jaxiga.utils.Solvers import Poisson2D_DEM_Spline
from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils.lbfgs import minimize as lbfgs_minimize
from jaxiga.utils.boundary import boundary2D, applyBC2D
from jaxiga.utils.processing_splines import (evaluate_spline_basis_fem_2d, 
                                      make_rhs,
                                      pde_form_poisson_2d,
                                      evaluate_stiff_rhs_fem_2d)
from jaxiga.utils_iga.assembly import gen_gauss_pts
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
def a0(x, y):
    return 1.0

def f(x, y):
    return (
        -2 * y ** 4
        + ((-48 * x ** 2 + 33) * y ** 2) / 2
        - 2 * x ** 4
        + (33 * x ** 2) / 2
        - 5
    )

def u_bound(x, y):
    return 0.0

# The exact solution for error norm computations
def exact_sol(x, y):
    return ((1 - x ** 2) * (1 - y ** 2) * (x ** 2 + y ** 2 - 1 / 4),)

def deriv_exact_sol(x, y):
    return [
        2 * x * y ** 4 + ((8 * x ** 3 - 9 * x) * y ** 2) / 2 - 4 * x ** 3 + (5 * x) / 2,
        ((8 * x ** 2 - 8) * y ** 3) / 2 + ((4 * x ** 4 - 9 * x ** 2 + 5) * y) / 2,
    ]
# Step 1: Generate the geometry
# Patch 1:
patch1 = PlateWHoleQuadrant(0.5, 1.0, 2)

# Patch 2:
patch2 = PlateWHoleQuadrant(0.5, 1.0, 3)

# Patch 3
patch3 = PlateWHoleQuadrant(0.5, 1.0, 4)

# Patch 4
patch4 = PlateWHoleQuadrant(0.5, 1.0, 1)

patches = [patch1, patch2, patch3, patch4]


# Set the boundary conditions
bound_outer_up_left = boundary2D("Dirichlet", 0, "up", u_bound)
bound_outer_down_left = boundary2D("Dirichlet", 1, "up", u_bound)
bound_outer_down_right = boundary2D("Dirichlet", 2, "up", u_bound)
bound_outer_up_right = boundary2D("Dirichlet", 3, "up", u_bound)
bound_inner_up_left = boundary2D("Dirichlet", 0, "down", u_bound)
bound_inner_down_left = boundary2D("Dirichlet", 1, "down", u_bound)
bound_inner_down_right = boundary2D("Dirichlet", 2, "down", u_bound)
bound_inner_up_right = boundary2D("Dirichlet", 3, "down", u_bound)
bound_cond = [
    bound_outer_up_left,
    bound_outer_down_left,
    bound_outer_down_right,
    bound_outer_up_right,
    bound_inner_up_left,
    bound_inner_down_left,
    bound_inner_down_right,
    bound_inner_up_right,
]

# Step 2: Degree elevate and refine the geometry
t = time.time()
for patch in patches:    
    patch.degreeElev(deg - 2, deg - 1)
elapsed = time.time() - t
print("Degree elevation took ", elapsed, " seconds")

t = time.time()
for i in range(num_refinements):
    for patch in patches:
        patch.refine_knotvectors(True, True)        
elapsed = time.time() - t
print("Knot insertion took ", elapsed, " seconds")

_, ax = plt.subplots()
for patch in patches:
    patch.plotKntSurf(ax)
plt.show()

t = time.time()
meshes = []
for patch in patches:
    meshes.append(IGAMesh2D(patch))
elapsed = time.time() - t
print("Mesh initialization took ", elapsed, " seconds")

for mesh in meshes:
    mesh.classify_boundary()
vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(meshes)
edge_list = gen_edge_list(patch2vertex)
size_basis = zip_conforming(meshes, vertex2patch, edge_list)
patch_list = [patch1]

# Generate the Gauss points        
# Evaluate the spline basis
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_quad_v = gen_gauss_pts(deg + 1)
gauss_rule = [gauss_quad_u, gauss_quad_v]
param_funs = (a0, f)
num_fields = 1
R, dR, local_areas, phys_pts, global_nodes_all = evaluate_spline_basis_fem_2d(meshes,
                                                                              gauss_rule,
                                                                              num_fields)

local_areas = jnp.reshape(local_areas, (-1, 1))
# Get the boundary degrees of freedom and set them to zeros
bcdof = get_bcdof(bound_cond, meshes)
bcval = jnp.zeros(len(bcdof))

model = Poisson2D_DEM_Spline(global_nodes_all, deg, size_basis, bcdof, bcval)

phys_pts_all = jnp.reshape(phys_pts, (-1, 2))
fvals = f(phys_pts_all[:, [0]], phys_pts_all[:, [1]])

num_basis = (model.deg+1)**2
model.train(R, dR, fvals, local_areas, n_iter = 1000)

t_start = time.time()
params = model.get_params(model.opt_state)
loss_func = jax_tfp_function_factory(model,  params, R, dR, fvals, local_areas)
#loss_func = jax_tfp_function_factory(model,  params, N_i, dNdx_i, dNdy_i, fvals, local_areas)
initial_pos = loss_func.init_params_1d
tolerance = 1e-6
current_loss, _ = loss_func(initial_pos)
print("Initial loss is ", current_loss)
num_bfgs_iterations = 0

results = lbfgs_minimize(loss_func, initial_position = initial_pos,
                         num_correction_pairs=100, max_iterations=2000)
initial_pos = results.position
num_bfgs_iterations += results.num_iterations
print("Iteration: ", num_bfgs_iterations, " loss: ", results.objective_value)
    
print("Time taken (BFGS) is ", time.time() - t_start)

_, unflatten_params = jax.flatten_util.ravel_pytree(params)
interior_vals = unflatten_params(results.position)

sol0 = jnp.zeros((size_basis))
sol0 = sol0.at[bcdof].set(bcval)
sol0 = sol0.at[model.trainable_indx].set(interior_vals[:,0])
sol0 = np.array(sol0)
t = time.time()
output_filename = "poisson_2d_square"
plot_sol2D(meshes, sol0, output_filename)
elapsed = time.time() - t
print("Plotting to VTK took ", elapsed, " seconds")

t = time.time()
# compute the solution at a set of uniformly spaced points
num_pts_xi = 100
num_pts_eta = 100
meas_vals_all, meas_pts_phys_xy_all, vals_min, vals_max = comp_measurement_values(num_pts_xi,
                                                              num_pts_eta,
                                                              meshes,
                                                              sol0,
                                                              get_measurements_vector,
                                                              num_fields)
elapsed = time.time() - t
print("Computing the values at measurement points took ", elapsed, " seconds")

t = time.time()
field_title = "Computed solution"
field_names = ['U']
plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes, 
                meas_pts_phys_xy_all, meas_vals_all, vals_min, vals_max)
elapsed = time.time() - t
print("Plotting the solution (matplotlib) took ", elapsed, " seconds")

# compute the error at a set of uniformy spaced points
t = time.time()
err_vals_all, err_vals_min, err_vals_max = comp_error_2D(num_fields, 
                                                          meshes,
                                                          exact_sol, 
                                                          meas_pts_phys_xy_all,
                                                          meas_vals_all)
elapsed = time.time() - t
print("Computing the error at measurement points took ", elapsed, " seconds")

# plot the error as a contour plot
t = time.time()
field_title = "Error in the solution"
plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes, 
                meas_pts_phys_xy_all, err_vals_all, err_vals_min, err_vals_max)
elapsed = time.time() - t
print("Plotting the error (matplotlib) took ", elapsed, " seconds")


# Compute the norm of the error
t = time.time()
rel_L2_err, rel_H1_err = comp_error_norm(meshes, sol0, exact_sol,
                                          deriv_exact_sol, a0,
                                          gauss_rule
)
print("Relative L2-norm error is ", rel_L2_err)
print("Relative energy-norm error is ", rel_H1_err)
elapsed = time.time() - t
print("Computing the error norms took", elapsed, " seconds")
