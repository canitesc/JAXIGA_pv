#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM with splines basis (i.e. IGA)

@author: cosmin
"""
import pypardiso
import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import config
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
# import cvxopt ;import cvxopt.cholmod


from jaxiga.utils.Geom_examples import Cuboid
from jaxiga.utils.IGA import IGAMesh3D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming

from jaxiga.utils.boundary3D import boundary3D, applyBC3D
from jaxiga.utils.processing_splines_3d import (evaluate_spline_basis_fem_3d, 
                                      make_rhs,
                                      pde_form_poisson_3d,
                                      evaluate_stiff_rhs_fem_3d)
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils_iga.postprocessing import (plot_sol3D,
                                      plot_sol3D_error,
                                      gen_param_weights_3d,
                                      get_physpts_3d,
                                      comp_error_norm_3d)

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 8
num_refinements = 2

# Step 0: Define the function and boundary conditions
def a0(x, y, z):
    return 1.0

def f(x, y, z):
    return 12 * jnp.pi ** 2 * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y) * \
        jnp.sin(2 * jnp.pi * z)
    # return -2*x*y*(x - 1)*(y - 1) - 2*x*z*(x - 1)*(z - 1) - 2*y*z*(y - 1)*(z - 1)
    
def u_bound(x, y, z):
    return 0.0

# The exact solution for error norm computations
def exact_sol(x, y, z):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)
    # return x*y*z*(x-1)*(y-1)*(z-1)

def deriv_exact_sol(x, y, z):
    return [
        2 * jnp.pi * jnp.cos(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y) * jnp.sin(2 * jnp.pi * z),
        2 * jnp.pi * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y) * jnp.sin(2 * jnp.pi * z),
        2 * jnp.pi * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y) * jnp.cos(2 * jnp.pi * z)
        # y*z*(2*x - 1)*(y - 1)*(z - 1),
        # x*z*(x - 1)*(2*y - 1)*(z - 1),
        # x*y*(x - 1)*(y - 1)*(2*z - 1)
    ]

bound_left = boundary3D("Dirichlet", 0, "left", u_bound)
bound_right = boundary3D("Dirichlet", 0, "right", u_bound)
bound_down = boundary3D("Dirichlet", 0, "down", u_bound)
bound_up = boundary3D("Dirichlet", 0, "up", u_bound)
bound_front = boundary3D("Dirichlet", 0, "front", u_bound)
bound_back = boundary3D("Dirichlet", 0, "back", u_bound)
bound_cond = [bound_left, bound_right, bound_down, bound_up, bound_front, bound_back]

# Generate the geometry
# Patch 1:
corners = jnp.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 1.],
                     [1., 0., 0.], [1., 1., 0.], [1., 0., 1.], [1., 1., 1.]])
patch1 = Cuboid(corners)

# Step 2: Degree elevate and refine the geometry
t = time.time()
patch1.degreeElev(deg - 1, deg - 1, deg-1)
# patch1.degreeElev(0, 1, 2)
elapsed = time.time() - t
print("Degree elevation took ", elapsed, " seconds")

t = time.time()
for i in range(num_refinements):
    patch1.refine_knotvectors(True, True, True)
elapsed = time.time() - t
print("Knot insertion took ", elapsed, " seconds")

mesh1 = IGAMesh3D(patch1)
mesh1.classify_boundary()
meshes = [mesh1]
# generate the Gauss points
num_patches = 1
num_gauss = 2
num_elems = [10, 10, 10]
meas_pts_param_i, weights_param = gen_param_weights_3d(num_patches, num_gauss, num_elems)

t = time.time()
Xint, Wint = get_physpts_3d(meshes, meas_pts_param_i, weights_param)
Yint = f(Xint[:,[0]], Xint[:,[1]], Xint[:,[2]])
print("Generating Gauss points took", time.time()-t, "seconds")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Xint[:,0], Xint[:,1], Xint[:,2], s=0.5, label='Gauss point')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.zaxis.labelpad=-0.7 # <- change the value here
plt.legend()
plt.show()
print('Volume of the domain is', np.sum(Wint))

time_start_process = time.time()
# Generate the Gauss points        
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_quad_v = gen_gauss_pts(deg + 1)
gauss_quad_w = gen_gauss_pts(deg + 1)
# Poor man's zip_conforming for one-patch geometry
meshes[0].elem_node_global = meshes[0].elem_node 
meshes[0].bcdof_global = meshes[0].bcdof
size_basis = meshes[0].num_basis
gauss_rule = [gauss_quad_u, gauss_quad_v, gauss_quad_w]
param_funs = (a0, f)
num_fields = 1
# Evaluate the spline basis
R, dR, local_areas, phys_pts, global_nodes_all = evaluate_spline_basis_fem_3d(meshes,
                                                                              gauss_rule,
                                                                              num_fields)
II, JJ, S, local_rhss = evaluate_stiff_rhs_fem_3d(R, dR, local_areas, phys_pts,
                                                  gauss_rule, num_fields,
                                                  global_nodes_all,
                                                  pde_form_poisson_3d,
                                                  param_funs, ())
t = time.time()
stiff_mat = sparse.coo_matrix((S, (II, JJ))).tocsr()
rhs = np.asarray(make_rhs(global_nodes_all, local_rhss, num_fields, size_basis))

stiff, rhs = applyBC3D(meshes, bound_cond, stiff_mat, rhs)
elapsed = time.time() - t
print("Applying B.C.s took ", elapsed, " seconds")

# Solve the linear system
t = time.time()
#sol0 = spsolve(stiff, rhs)
sol0 = pypardiso.spsolve(stiff, rhs)
#sol0 = cvxopt.cholmod.linsolve(stiff, rhs)
elapsed = time.time() - t
print("Linear sparse solver took ", elapsed, " seconds")

time_end_process = time.time()
print("Total time for processing ", time_end_process - time_start_process)

t = time.time()
output_filename = "poisson_3d_cube"
plot_sol3D(meshes, sol0, output_filename)
plot_sol3D_error(meshes, sol0, exact_sol, output_filename+"_err")
elapsed = time.time() - t
print("Plotting to VTK took ", elapsed, " seconds")

# Compute the norm of the error
t = time.time()
rel_L2_err, rel_H1_err = comp_error_norm_3d(meshes, sol0, exact_sol,
                                          deriv_exact_sol, a0,
                                          gauss_rule
)
print("Relative L2-norm error is ", rel_L2_err)
print("Relative energy-norm error is ", rel_H1_err)
elapsed = time.time() - t
print("Computing the error norms took", elapsed, " seconds")
