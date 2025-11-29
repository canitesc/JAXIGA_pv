#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:02:13 2024

Simulate the elastic plate with holes example on a Cartesian grid

@author: cosmin
"""
import jax
import pypardiso
# import cvxopt
# import cvxopt.cholmod
# from cvxopt import spmatrix

import jax.flatten_util
import jax.numpy as jnp
from jax import config
import matplotlib.tri as tri
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from jaxiga.utils.Geom import Geometry2D
from jaxiga.utils.Geom_examples import Quadrilateral
from jaxiga.utils_iga.materials import MaterialElast2D

from jaxiga.utils.IGA import IGAMesh2D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming

from jaxiga.utils.boundary import boundary2D, applyBCElast2D
from jaxiga.utils.processing_splines import (evaluate_spline_basis_fem_2d,
                                      make_rhs,
                                      pde_form_elast_2d_comp,
                                      evaluate_stiff_rhs_fem_2d)
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils_iga.postprocessing import (plot_sol2D_elast,
                                      plot_sol2D_elast_error,
                                      plot_fields_2D,
                                      comp_error_2D,
                                      exact_stress_vect,
                                      comp_measurement_values,
                                      get_measurements_vector,
                                      get_measurement_stresses_composite,
                                      comp_error_norm_elast)

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 1#5
num_refinements = 8#4
output_filename = "elast2d_plate_w_five_holes_cart"


def create_matrix_with_inclusions(phys_corners, phys_inclusions, nx, ny):
    xmin, ymin, xmax, ymax = phys_corners
    matrix = np.ones((ny, nx))
    
    # Calculate the scaling factors to convert physical coordinates to matrix coordinates
    x_scale = (nx - 1) / (xmax - xmin)
    y_scale = (ny - 1) / (ymax - ymin)
    
    for (xc, yc, r) in phys_inclusions:
        for i in range(nx):
            for j in range(ny):
                # Calculate the physical coordinates of the matrix cell
                x_phys = xmin + i / x_scale
                y_phys = ymin + j / y_scale
                
                # Check if the point (i, j) in the matrix falls within any of the inclusions
                if ((x_phys - xc) ** 2 / r ** 2 + (y_phys - yc) ** 2 / r ** 2) <= 1:
                    matrix[j, i] = 0
                    
    return matrix

def get_Emod_inclusions(phys_pts, Emod_mat, Emod_inc, phys_inclusions):    
    x = phys_pts[:, :, 0]
    y = phys_pts[:, :, 1]
    Emods = Emod_mat * jnp.ones_like(x)
    for (xc, yc, r) in phys_inclusions:
        Emods = jnp.where((x-xc)**2 + (y-yc)**2 <= r**2, Emod_inc, Emods)
    return Emods
        

# Example usage

nx = 128
ny = 128
phys_corners = [-6, -6, 18, 18]
phys_inclusions = [(0., 0., 3), (12., 0., 0.6), (6., 12., 1.), (-3., 9., 0.5),
                   (15, 15, 0.5)]

matrix = create_matrix_with_inclusions(phys_corners, phys_inclusions, nx, ny)
plt.imshow(matrix, origin='lower')
plt.colorbar()
plt.show()

# Step 0: Define the material properties
Emod_mat = 210e3  # Matrix inclusion
Emod_inc = 1e-6     # 
nu = 0.3

# symmetry Dirichlet B.C., u_y = 0 for y=0 and u_x=0 for x=0
def u_bound_dir_symy0(x, y):
    return [None, 0.0]

def u_bound_dir_symx0(x, y):
    return [0.0, None]

#  Neumann B.C. τ(x,y) = [x, y] on the circular boundary
bound_trac = 10.0  # traction at infinity in the x direction

def u_bound_neu(x, y, nx, ny):
    tau_x = 0.
    tau_y = bound_trac
    return [tau_x, tau_y]

quad_dom = np.array([[phys_corners[0], phys_corners[1]],
                       [phys_corners[0], phys_corners[3]],
                       [phys_corners[2], phys_corners[1]],
                       [phys_corners[2], phys_corners[3]]])
patch = Quadrilateral(quad_dom)
patches = [patch]

bound_up = boundary2D("Neumann", 0, "up", u_bound_neu)
bound_down = boundary2D("Dirichlet", 0, "down", u_bound_dir_symy0)
bound_left = boundary2D("Dirichlet", 0, "left", u_bound_dir_symx0)

bound_cond = [bound_up, bound_down, bound_left]

# Step 2: Degree elevate and refine the geometry
t = time.time()
for patch in patches:
    patch.degreeElev(deg - 1, deg - 1)
elapsed = time.time() - t
print("Degree elevation took ", elapsed, " seconds")

t = time.time()
for patch in patches:
    for i in range(num_refinements):
        patch.refine_knotvectors(True, True)

elapsed = time.time() - t
print("Knot insertion took ", elapsed, " seconds")

_, ax = plt.subplots()
for patch in patches:
    patch.plotKntSurf(ax)
plt.show()

t = time.time()
mesh_list =[]
for patch in patches:
    mesh = IGAMesh2D(patch)
    mesh.classify_boundary()
    mesh_list.append(mesh)
elapsed = time.time() - t
print("Mesh initialization took ", elapsed, " seconds")

vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(mesh_list)
edge_list = gen_edge_list(patch2vertex)
size_basis = zip_conforming(mesh_list, vertex2patch, edge_list)

# Generate the Gauss points        
# Evaluate the spline basis
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_quad_v = gen_gauss_pts(deg + 1)
gauss_rule = [gauss_quad_u, gauss_quad_v]
num_fields = 2
R, dR, local_areas, phys_pts, global_nodes_all = evaluate_spline_basis_fem_2d(mesh_list,
                                                                              gauss_rule,
                                                                              num_fields)

Emods = get_Emod_inclusions(phys_pts, Emod_mat, Emod_inc, phys_inclusions)
phys_pts_plot = jnp.reshape(phys_pts, (-1, 2))
triang = tri.Triangulation(phys_pts_plot[:, 0], phys_pts_plot[:, 1])
plt.tricontourf(triang, Emods.flatten())
plt.colorbar()
plt.axis('equal')
plt.title('Emod')
plt.show()

II, JJ, S, local_rhss = evaluate_stiff_rhs_fem_2d(R, dR, local_areas, phys_pts,
                                                  gauss_rule, num_fields,
                                                  global_nodes_all,
                                                  pde_form_elast_2d_comp,
                                                  (nu,), (Emods,))
t = time.time()
stiff_mat = sparse.coo_matrix((S, (II, JJ))).tocsr()

# def scipy_sparse_to_spmatrix(A):
#     coo = A.tocoo()
#     SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
#     return SP

rhs = np.asarray(make_rhs(global_nodes_all, local_rhss, num_fields, size_basis)).copy()
stiff, rhs = applyBCElast2D(mesh_list, bound_cond, stiff_mat, rhs, gauss_rule)
elapsed = time.time() - t
print("Applying B.C.s took ", elapsed, " seconds")

# Solve the linear system
t = time.time()
# # stiff = scipy_sparse_to_spmatrix(stiff)
# stiff = stiff.tocoo()
# print("Converting stiff to COO took", time.time()-t, "seconds")
# t = time.time()
# stiff = spmatrix(stiff.data, stiff.row, stiff.col, size=stiff.shape)
# print("Converting stiff to spmatrix took", time.time()-t, "seconds")
t = time.time()
sol0 = pypardiso.spsolve(stiff, rhs)
# sol0 = cvxopt.cholmod.linsolve(stiff, rhs)
# sol0 = spsolve(stiff, rhs)
elapsed = time.time() - t
print("Linear sparse solver took ", elapsed, " seconds")

t = time.time()
# compute the displacements at a set of uniformly spaced points
num_pts_xi = 25
num_pts_eta = 25
meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = comp_measurement_values(num_pts_xi,
                                                              num_pts_eta,
                                                              mesh_list,
                                                              sol0,
                                                              get_measurements_vector,
                                                              num_fields)
elapsed = time.time() - t
print("Computing the displacement values at measurement points took ", elapsed, " seconds")


# t = time.time()
# # compute the stresses at a set of uniformly spaced points
# num_output_fields = 4
# meas_stress_all, meas_pts_phys_xy_all, vals_stress_min, vals_stress_max = comp_measurement_values(num_pts_xi,
#                                                               num_pts_eta,
#                                                               mesh_list,
#                                                               sol0,
#                                                               get_measurement_stresses_composite,
#                                                               num_output_fields,
#                                                               Emods,
#                                                               nu)
# elapsed = time.time() - t
# print("Computing the stress values at measurement points took ", elapsed, " seconds")

t = time.time()
field_title = "Computed solution"
field_names = ['x-disp', 'y-disp']
plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list, 
                meas_pts_phys_xy_all, meas_vals_all, vals_disp_min, vals_disp_max)
elapsed = time.time() - t
print("Plotting the solution (matplotlib) took ", elapsed, " seconds")


