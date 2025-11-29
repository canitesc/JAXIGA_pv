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

from jaxiga.utils.Geom_examples import Quadrilateral
from jaxiga.utils_iga.materials import MaterialElast2D
from jaxiga.utils.IGA import IGAMesh2D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming
from jaxiga.utils.Solvers import Elast2D_DEM_Spline
from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils.lbfgs import minimize as lbfgs_minimize
from jaxiga.utils.boundary import boundary2D, get_bcdof_bcval, applyBCElast_DEM_2D
from jaxiga.utils.processing_splines import evaluate_spline_basis_fem_2d                                    
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils_iga.postprocessing import (plot_sol2D_elast,
                                      comp_measurement_values,
                                      get_measurements_vector,
                                      plot_fields_2D)

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 1
num_refinements = 8


# Step 0: Define the material properties
Emod = 1#e5
nu = 0#.3
material = MaterialElast2D(Emod=Emod, nu=nu, plane_type="stress")

# symmetry Dirichlet B.C., u_y = 0 for y=0 and u_x=0 for x=0


def u_bound_dir_fixed(x, y):
    return [0.0, 0.0]


def u_bound_dir_vert_disp(x, y, disp=0.1):
    return [0.0, 0.2]


# Set the boundary conditions
bound_down = boundary2D("Dirichlet", 0, "down", u_bound_dir_fixed)
bound_up = boundary2D("Dirichlet", 0, "up", u_bound_dir_vert_disp)
bound_cond = [bound_down, bound_up]

# Step 1: Generate the geometry
# Patch 1:
_, ax = plt.subplots()
plate_length = 1.
plate_width = 1.
vertices = [[0., 0.], [0., plate_width], [
    plate_length, 0.], [plate_length, plate_width]]
patch1 = Quadrilateral(np.array(vertices))
patch1.plotKntSurf(ax)
patches = [patch1]

# Step 2: Degree elevate and refine the geometry
t = time.time()
for patch in patches:
    patch.degreeElev(deg - 1, deg - 1)
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
num_fields = 2
R, dR, local_areas, phys_pts, global_nodes_all = evaluate_spline_basis_fem_2d(meshes,
                                                                              gauss_rule,
                                                                              num_fields)

local_areas = jnp.reshape(local_areas, (-1, 1))

# Get the boundary degrees of freedom 
bcdof, bcval = get_bcdof_bcval(meshes, bound_cond)

model = Elast2D_DEM_Spline(global_nodes_all, deg, size_basis, bcdof, bcval, material)

phys_pts_all = jnp.reshape(phys_pts, (-1, 2))

num_basis = (model.deg+1)**2

rhs = applyBCElast_DEM_2D(meshes, bound_cond, size_basis, gauss_rule, model.index_map)
# get the rhs for Neumann boundary conditions
rhs = applyBCElast_DEM_2D(meshes, bound_cond, size_basis, gauss_rule, model.index_map)
inv_index_map = jnp.argsort(model.index_map)
rhs = rhs[inv_index_map]
model.train(R, dR, jnp.squeeze(local_areas), rhs, n_iter=1000)

t_start = time.time()
params = model.get_params(model.opt_state)
loss_func = jax_tfp_function_factory(model,  params, R, dR, jnp.squeeze(local_areas), rhs)
initial_pos = loss_func.init_params_1d
tolerance = 1e-6
current_loss, _ = loss_func(initial_pos)
print("Initial loss is ", current_loss)
num_bfgs_iterations = 0

# results = bfgs_minimize(loss_func, initial_position = initial_pos,
#                     max_iterations=2000)
results = lbfgs_minimize(loss_func, initial_position=initial_pos,
                          num_correction_pairs=100, max_iterations=2000)
initial_pos = results.position
num_bfgs_iterations += results.num_iterations
print("Iteration: ", num_bfgs_iterations, " loss: ", results.objective_value)

print("Time taken (BFGS) is ", time.time() - t_start)

_, unflatten_params = jax.flatten_util.ravel_pytree(params)
interior_vals = unflatten_params(results.position)

sol0 = jnp.zeros((2*size_basis))
sol0 = sol0.at[bcdof].set(bcval)
sol0 = sol0.at[model.trainable_indx].set(interior_vals[:, 0])
sol0 = np.array(sol0)
t = time.time()
output_filename = "elast_2d_tension_plate_DEM"
plot_sol2D_elast(meshes, material.Cmat, sol0, output_filename)
elapsed = time.time() - t
print("Plotting to VTK took ", elapsed, " seconds")

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
field_names = ['x-disp', 'y-disp']
plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes, 
                meas_pts_phys_xy_all, meas_vals_all, vals_min, vals_max)
elapsed = time.time() - t
print("Plotting the solution (matplotlib) took ", elapsed, " seconds")
