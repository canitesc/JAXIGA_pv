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

from jaxiga.utils.Geom_examples import Quadrilateral
from jaxiga.utils.IGA import IGAMesh2D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming

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

deg = 2
num_refinements = 5

# Step 0: Define the function and boundary conditions
def a0(x, y):
    return 1.0

def f(x, y):
    return 8 * jnp.pi ** 2 * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)

def u_bound(x, y):
    return 0.0

# The exact solution for error norm computations
def exact_sol(x, y):
    return (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y),)

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
elapsed = time.time() - t
print("Knot insertion took ", elapsed, " seconds")

_, ax = plt.subplots()
patch1.plotKntSurf(ax)
plt.show()

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

# Generate the Gauss points        
# Evaluate the spline basis
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_quad_v = gen_gauss_pts(deg + 1)
gauss_rule = [gauss_quad_u, gauss_quad_v]
param_funs = (a0, f)
num_fields = 1
R, dR, local_areas, phys_pts, global_nodes_all = evaluate_spline_basis_fem_2d(mesh_list,
                                                                              gauss_rule,
                                                                              num_fields)
II, JJ, S, local_rhss = evaluate_stiff_rhs_fem_2d(R, dR, local_areas, phys_pts,
                                                  gauss_rule, num_fields,
                                                  global_nodes_all,
                                                  pde_form_poisson_2d,
                                                  param_funs, ())
t = time.time()
stiff_mat = sparse.coo_matrix((S, (II, JJ))).tocsr()
rhs = np.asarray(make_rhs(global_nodes_all, local_rhss, num_fields, size_basis))

stiff, rhs = applyBC2D(mesh_list, bound_cond, stiff_mat, rhs)
elapsed = time.time() - t
print("Applying B.C.s took ", elapsed, " seconds")

# Solve the linear system
t = time.time()
sol0 = spsolve(stiff, rhs)
elapsed = time.time() - t
print("Linear sparse solver took ", elapsed, " seconds")

t = time.time()
output_filename = "poisson_2d_square"
plot_sol2D(mesh_list, sol0, output_filename)
elapsed = time.time() - t
print("Plotting to VTK took ", elapsed, " seconds")

t = time.time()
# compute the solution at a set of uniformly spaced points
num_pts_xi = 100
num_pts_eta = 100
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
rel_L2_err, rel_H1_err = comp_error_norm(mesh_list, sol0, exact_sol,
                                         deriv_exact_sol, a0,
                                         gauss_rule
)
print("Relative L2-norm error is ", rel_L2_err)
print("Relative energy-norm error is ", rel_H1_err)
elapsed = time.time() - t
print("Computing the error norms took", elapsed, " seconds")
