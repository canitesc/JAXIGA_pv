#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poisson 1D example with Deep Energy Method

@author: cosmin
"""

import jax
import jax.flatten_util
import jax.numpy as jnp
import time

from tqdm import trange
import matplotlib.pyplot as plt
from jax import config
from scipy import sparse
import numpy as np
from scipy.sparse.linalg import spsolve

from jaxiga.utils.boundary import boundary1D, applyBC1D
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils.Geom_examples import Segment
from jaxiga.utils.IGA import IGAMesh1D
from jaxiga.utils.processing_splines_1d import (evaluate_spline_basis_fem_1d,
                                         evaluate_stiff_rhs_fem_1d,
                                         pde_form_poisson_1d,
                                         make_rhs)
from jaxiga.utils_iga.postprocessing_1d import (comp_measurement_values_1d,
                                      get_measurements_vector_1d)


key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

# define the problem
k = 1

# Step 0: Define the function and boundary conditions
def a0(x):
    return 1.0

exact_sol = lambda x : jnp.sin(k*jnp.pi*x)
rhs_fun = lambda x : k**2*jnp.pi**2*jnp.sin(k*jnp.pi*x)

u_bound = lambda x: 0.

bound_left = boundary1D("Dirichlet", 0, "left", u_bound)
bound_right = boundary1D("Dirichlet", 0, "right", u_bound)
bound_cond = [bound_left, bound_right]

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 2
num_refinements = 7

# Generate the geometry
# Patch 1:
endpoints = [[0.0], [8.0]]
patch1 = Segment(endpoints)

# Step 2: Degree elevate and refine the geometry
t = time.time()
patch1.degreeElev(deg - 1)
elapsed = time.time() - t
print("Degree elevation took ", elapsed, " seconds")

t = time.time()
for i in range(num_refinements):
    patch1.refine_knotvectors(True)
elapsed = time.time() - t
print("Knot insertion took ", elapsed, " seconds")

pts = patch1.getUnifIntPts(101, [1,1])
mesh1 = IGAMesh1D(patch1)
mesh1.classify_boundary()
meshes = [mesh1]
# Poor man's zip_conforming for one-patch geometry
meshes[0].elem_node_global = meshes[0].elem_node 
meshes[0].bcdof_global = meshes[0].bcdof
size_basis = meshes[0].num_basis

# Evaluate the spline basis
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_rule = [gauss_quad_u]
num_fields = 1

param_funs = (a0, rhs_fun)

R, dR, local_areas, phys_pts, global_nodes_all = evaluate_spline_basis_fem_1d(meshes,
                                                                              gauss_rule,
                                                                              num_fields)

II, JJ, S, local_rhss = evaluate_stiff_rhs_fem_1d(R, dR, local_areas, phys_pts,
                                                  gauss_rule, num_fields,
                                                  global_nodes_all,
                                                  pde_form_poisson_1d,
                                                  param_funs, ())

t = time.time()
stiff_mat = sparse.coo_matrix((S, (II, JJ))).tocsr()
rhs = np.asarray(make_rhs(global_nodes_all, local_rhss, num_fields, size_basis))

stiff, rhs = applyBC1D(meshes, bound_cond, stiff_mat, rhs)
elapsed = time.time() - t
print("Applying B.C.s took ", elapsed, " seconds")

# Solve the linear system
t = time.time()
sol0 = spsolve(stiff, rhs)
elapsed = time.time() - t
print("Linear sparse solver took ", elapsed, " seconds")


# compute the solution at a set of uniformly spaced points
num_pts_xi = 1001
meas_vals_all, meas_pts_phys_x_all, vals_min, vals_max = comp_measurement_values_1d(num_pts_xi,
                                                              meshes,
                                                              sol0,
                                                              get_measurements_vector_1d,
                                                              num_fields)
elapsed = time.time() - t
print("Computing the values at measurement points took ", elapsed, " seconds")
x_pts = meas_pts_phys_x_all[0].flatten()
u_comp = meas_vals_all[0][0].flatten()
plt.plot(x_pts, u_comp)
plt.title('Computed solution')
plt.show()
u_exact = exact_sol(meas_pts_phys_x_all[0]).flatten()
plt.plot(x_pts, u_exact-u_comp)
plt.title('Error for the computed solution')
plt.show()

print("L2-error norm: {}".format(np.linalg.norm(u_exact-u_comp)/np.linalg.norm(u_exact)))
