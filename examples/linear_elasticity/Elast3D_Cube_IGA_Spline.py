#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM with splines basis (i.e. IGA)

Elasticity 3D Cube example

@author: cosmin
"""
#import pypardiso
import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import config
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import cvxopt ;import cvxopt.cholmod


from jaxiga.utils.Geom_examples import Cuboid
from jaxiga.utils.IGA import IGAMesh3D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming

from jaxiga.utils.boundary3D import boundary3D, applyBCElast3D
from jaxiga.utils.processing_splines_3d import (evaluate_spline_basis_fem_3d, 
                                      make_rhs,
                                      pde_form_elast_3d,
                                      evaluate_stiff_rhs_fem_3d)
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils_iga.materials import MaterialElast3D
from jaxiga.utils_iga.postprocessing import (plot_sol3D_elast,
                                      plot_sol3D_error,
                                      gen_param_weights_3d,
                                      get_physpts_3d,
                                      cart2sph,
                                      sph2cart,
                                      comp_error_norm_3d)

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 2
num_refinements = 3

pressure = 10.
side = 1.
Emod = 1000.
nu = 0.3
material = MaterialElast3D(Emod=Emod, nu=nu)


# Step 0: Define the function and boundary conditions
# symmetry Dirichlet B.C., u_z = 0 for z=0, u_y = 0 for y=0, and u_x=0 for x=0

def u_bound_dir_symx0(x, y, z):
    return [0., None, None]

def u_bound_dir_symy0(x, y, z):
    return [None, 0., None]

def u_bound_dir_symz0(x, y, z):
    return [None, None, 0.0]

def u_bound_neu(x, y, z, nx, ny, nz):
    return [nx*pressure, ny*pressure, nz*pressure]


bound_left = boundary3D("Dirichlet", 0, "left", u_bound_dir_symx0)
bound_front = boundary3D("Dirichlet", 0, "front", u_bound_dir_symy0)
bound_down = boundary3D("Dirichlet", 0, "down", u_bound_dir_symz0)
bound_up = boundary3D("Neumann", 0, "up", u_bound_neu)
# bound_left = boundary3D("Dirichlet", 0, "down", u_bound_dir_symx0)
# bound_front = boundary3D("Dirichlet", 0, "left", u_bound_dir_symy0)
# bound_down = boundary3D("Dirichlet", 0, "front", u_bound_dir_symz0)
# bound_up = boundary3D("Neumann", 0, "back", u_bound_neu)
bound_cond = [bound_left, bound_front, bound_down, bound_up]

# Generate the geometry
# Patch 1:
# corners = jnp.array([[0., 0., 0.], [0., side, 0.], [0., 0., side], [0., side, side],
#                      [side, 0., 0.], [side, side, 0.], [side, 0., side], [side, side, side]])
corners = jnp.array([[0., 0., 0.], [side, 0., 0.], [0., side, 0.], [side, side, 0.],
                     [0., 0., side], [side, 0., side], [0, side, side], [side, side, side]])
patch1 = Cuboid(corners)

# coefs = patch1.getCoefs()
# patch1.setCoefs(coefs)

# Step 2: Degree elevate and refine the geometry
t = time.time()
patch1.degreeElev(deg - 1, deg - 1, deg-1)
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
num_elems = [8, 8, 8]
meas_pts_param_i, weights_param = gen_param_weights_3d(num_patches, num_gauss, num_elems)

t = time.time()
Xint, Wint = get_physpts_3d(meshes, meas_pts_param_i, weights_param)
print("Generating Gauss points took", time.time()-t, "seconds")

x_bnd_front, y_bnd_front, z_bnd_front, _, _, _, _ = patch1.getQuadFacePts(num_elems, num_gauss+2, 1)
x_bnd_back, y_bnd_back, z_bnd_back, _, _, _, _ = patch1.getQuadFacePts(num_elems, num_gauss+2, 3)
x_bnd_right, y_bnd_right, z_bnd_right, _, _, _, _ = patch1.getQuadFacePts(num_elems, num_gauss+2, 2)
x_bnd_left, y_bnd_left, z_bnd_left, _, _, _, _ = patch1.getQuadFacePts(num_elems, num_gauss+2, 4)
x_bnd_down, y_bnd_down, z_bnd_down, _, _, _, _ = patch1.getQuadFacePts(num_elems, num_gauss+2, 5)
x_bnd_up, y_bnd_up, z_bnd_up, _, _, _, _ = patch1.getQuadFacePts(num_elems, num_gauss+2, 6)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Xint[:, 0], Xint[:, 1], Xint[:, 2], s=0.5, label='Gauss point')
ax.scatter(x_bnd_front, y_bnd_front, z_bnd_front, s=0.5, label='Front')
ax.scatter(x_bnd_back, y_bnd_back, z_bnd_back, s=0.5, label='Back')
ax.scatter(x_bnd_right, y_bnd_right, z_bnd_right, s=0.5, label='Right')
ax.scatter(x_bnd_left, y_bnd_left, z_bnd_left, s=0.5, label='Left')
ax.scatter(x_bnd_down, y_bnd_down, z_bnd_down, s=0.5, label='Down')
ax.scatter(x_bnd_up, y_bnd_up, z_bnd_up, s=0.5, label='Up')
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
num_fields = 3
# Evaluate the spline basis
R, dR, local_areas, phys_pts, global_nodes_all = evaluate_spline_basis_fem_3d(meshes,
                                                                              gauss_rule,
                                                                              num_fields)
II, JJ, S, local_rhss = evaluate_stiff_rhs_fem_3d(R, dR, local_areas, phys_pts,
                                                  gauss_rule, num_fields,
                                                  global_nodes_all,
                                                  pde_form_elast_3d,
                                                  (material,), ())
t = time.time()
stiff_mat = sparse.coo_matrix((S, (II, JJ))).tocsr()
rhs = np.array(make_rhs(global_nodes_all, local_rhss, num_fields, size_basis))

stiff, rhs, phys_pts, normals = applyBCElast3D(meshes, bound_cond, stiff_mat, rhs, gauss_rule)
elapsed = time.time() - t
print("Applying B.C.s took ", elapsed, " seconds")

# Solve the linear system
t = time.time()
sol0 = spsolve(stiff, rhs)
#sol0 = pypardiso.spsolve(stiff, rhs)
#sol0 = cvxopt.cholmod.linsolve(stiff, rhs)
elapsed = time.time() - t
print("Linear sparse solver took ", elapsed, " seconds")

time_end_process = time.time()
print("Total time for processing ", time_end_process - time_start_process)

t = time.time()
output_filename = "elast_3d_cube"
plot_sol3D_elast(meshes, material.Cmat, sol0, output_filename)
#plot_sol3D_error(meshes, sol0, exact_sol, output_filename+"_err")
elapsed = time.time() - t
print("Plotting to VTK took ", elapsed, " seconds")

# # Compute the norm of the error
# t = time.time()
# rel_L2_err, rel_H1_err = comp_error_norm_3d(meshes, sol0, exact_sol,
#                                           deriv_exact_sol, a0,
#                                           gauss_rule
# )
# print("Relative L2-norm error is ", rel_L2_err)
# print("Relative energy-norm error is ", rel_H1_err)
# elapsed = time.time() - t
# print("Computing the error norms took", elapsed, " seconds")
