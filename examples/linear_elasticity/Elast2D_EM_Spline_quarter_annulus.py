#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve a 2D elasticity problem on a quarter annulus domain

\Omega = {(x,y) : r_int^2 < x^2+y^2 < r_ext^2, x>0, y>0}

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

from jaxiga.utils.Geom_examples import QuarterAnnulus
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
                                      get_measurement_stresses,
                                      plot_fields_2D,
                                      comp_error_2D,
                                      comp_error_norm_elast,
                                      exact_stress_vect)

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 5
num_refinements = 5

# Step 0: Define the material properties
Emod = 1e5
nu = 0.3
material = MaterialElast2D(Emod=Emod, nu=nu, plane_type="stress")

# symmetry Dirichlet B.C., u_y = 0 for y=0 and u_x=0 for x=0
def u_bound_dir_symy0(x, y):
    return [None, 0.0]


def u_bound_dir_symx0(x, y):
    return [0.0, None]

# pressure Neumann B.C. τ(x,y) = [x, y] on the circular boundary
bound_press = 10.0


def u_bound_neu(x, y, nx, ny):
    return [-bound_press * nx, -bound_press * ny]


# The exact solution for error norm computations
def exact_sol(x, y):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    u_r = (
        rad_int ** 2
        * bound_press
        * r
        / (Emod * (rad_ext ** 2 - rad_int ** 2))
        * (1 - nu + (rad_ext / r) ** 2 * (1 + nu))
    )
    return [u_r * np.cos(th), u_r * np.sin(th)]

def _exact_stress(x, y):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    sigma_rr = (
        rad_int ** 2
        * bound_press
        / (rad_ext ** 2 - rad_int ** 2)
        * (1 - rad_ext ** 2 / r ** 2)
    )
    sigma_tt = (
        rad_int ** 2
        * bound_press
        / (rad_ext ** 2 - rad_int ** 2)
        * (1 + rad_ext ** 2 / r ** 2)
    )
    sigma_rt = 0.0

    A = np.array(
        [
            [np.cos(th) ** 2, np.sin(th) ** 2, 2 * np.sin(th) * np.cos(th)],
            [np.sin(th) ** 2, np.cos(th) ** 2, -2 * np.sin(th) * np.cos(th)],
            [
                -np.sin(th) * np.cos(th),
                np.sin(th) * np.cos(th),
                np.cos(th) ** 2 - np.sin(th) ** 2,
            ],
        ]
    )
    stress = np.linalg.solve(A, [sigma_rr, sigma_tt, sigma_rt])
    return stress

exact_stress = lambda x, y : exact_stress_vect(x, y, _exact_stress)

# Set the boundary conditions
bound_left = boundary2D("Neumann", 0, "left", u_bound_neu)
bound_down = boundary2D("Dirichlet", 0, "down", u_bound_dir_symy0)
bound_up = boundary2D("Dirichlet", 0, "up", u_bound_dir_symx0)
bound_cond = [bound_left, bound_down, bound_up]

# Step 1: Generate the geometry
# Patch 1:
_, ax = plt.subplots()
rad_int = 1.0
rad_ext = 4.0
patch = QuarterAnnulus(rad_int, rad_ext)
patches = [patch]

# Step 2: Degree elevate and refine the geometry
t = time.time()
patch.degreeElev(deg - 1, deg - 2)
elapsed = time.time() - t
print("Degree elevation took ", elapsed, " seconds")

t = time.time()
for i in range(num_refinements):
    patch.refine_knotvectors(True, True)

elapsed = time.time() - t
print("Knot insertion took ", elapsed, " seconds")

_, ax = plt.subplots()
for patch in patches:
    patch.plotKntSurf(ax)
plt.show()

t = time.time()
mesh = IGAMesh2D(patch)
elapsed = time.time() - t
print("Mesh initialization took ", elapsed, " seconds")

mesh.classify_boundary()
meshes = [mesh]
vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(meshes)
edge_list = gen_edge_list(patch2vertex)
size_basis = zip_conforming(meshes, vertex2patch, edge_list)


# Generate the Gauss points
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_quad_v = gen_gauss_pts(deg + 1)
gauss_rule = [gauss_quad_u, gauss_quad_v]
# Evaluate the spline basis
num_fields = 2
R, dR, local_areas, phys_pts, global_nodes_all = evaluate_spline_basis_fem_2d(meshes,
                                                                              gauss_rule,
                                                                              num_fields)

local_areas = jnp.reshape(local_areas, (-1, 1))

# Get the boundary degrees of freedom and set them to zeros
bcdof, bcval = get_bcdof_bcval(meshes, bound_cond)

model = Elast2D_DEM_Spline(global_nodes_all, deg, size_basis, bcdof, bcval, material)

phys_pts_all = jnp.reshape(phys_pts, (-1, 2))

num_basis = (model.deg+1)**2

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
output_filename = "elast_2d_quarter_annulus_DEM"
plot_sol2D_elast(meshes, material.Cmat, sol0, output_filename)
elapsed = time.time() - t
print("Plotting to VTK took ", elapsed, " seconds")

t = time.time()
# compute the displacements at a set of uniformly spaced points
num_pts_xi = 100
num_pts_eta = 100
meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = comp_measurement_values(num_pts_xi,
                                                              num_pts_eta,
                                                              meshes,
                                                              sol0,
                                                              get_measurements_vector,
                                                              num_fields)
elapsed = time.time() - t
print("Computing the displacement values at measurement points took ", elapsed, " seconds")

t = time.time()
# compute the stresses at a set of uniformly spaced points
num_pts_xi = 100
num_output_fields = 4
meas_stress_all, meas_pts_phys_xy_all, vals_stress_min, vals_stress_max = comp_measurement_values(num_pts_xi,
                                                              num_pts_eta,
                                                              meshes,
                                                              sol0,
                                                              get_measurement_stresses,
                                                              num_output_fields,
                                                              material.Cmat)
elapsed = time.time() - t
print("Computing the stress values at measurement points took ", elapsed, " seconds")


t = time.time()
field_title = "Computed solution"
field_names = ['x-disp', 'y-disp']
plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes, 
                meas_pts_phys_xy_all, meas_vals_all, vals_disp_min, vals_disp_max)
elapsed = time.time() - t
print("Plotting the solution (matplotlib) took ", elapsed, " seconds")

t = time.time()
field_title = "Computed solution"
field_names = ['xx-stress', 'yy-stress', 'xy-stress', 'VM-stress']
plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes, 
                meas_pts_phys_xy_all, meas_stress_all, vals_stress_min, vals_stress_max)
elapsed = time.time() - t
print("Plotting the solution (matplotlib) took ", elapsed, " seconds")


err_vals_all, err_vals_min, err_vals_max = comp_error_2D(num_fields, 
                                                          meshes,
                                                          exact_sol, 
                                                          meas_pts_phys_xy_all,
                                                          meas_vals_all)

stress_err_vals_all, stress_err_vals_min, stress_err_vals_max = comp_error_2D(num_output_fields, 
                                                                              meshes,
                                                                              exact_stress,
                                                                              meas_pts_phys_xy_all,
                                                                              meas_stress_all)

elapsed = time.time() - t
print("Computing the error at measurement points took ", elapsed, " seconds")

# plot the error as a contour plot
t = time.time()
field_title = "Error in the solution"
field_names = ['x-disp', 'y-disp']
plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes, 
                meas_pts_phys_xy_all, err_vals_all, err_vals_min, err_vals_max)
field_names = ['xx-stress', 'yy-stress', 'xy-stress', 'VM-stress']
plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes, 
                meas_pts_phys_xy_all, stress_err_vals_all, stress_err_vals_min,
                stress_err_vals_max)
elapsed = time.time() - t
print("Plotting the error (matplotlib) took ", elapsed, " seconds")


# Compute the norm of the error
t = time.time()
rel_L2_err, rel_H1_err = comp_error_norm_elast(
    meshes, material.Cmat, sol0, exact_sol, exact_stress, gauss_rule
)
print("Relative L2-norm error is ", rel_L2_err)
print("Relative energy-norm error is ", rel_H1_err)
elapsed = time.time() - t
print("Computing the error norms took", elapsed, " seconds")