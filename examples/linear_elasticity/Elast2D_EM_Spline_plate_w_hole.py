#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve a 2D elasticity problem on a plate with a void domain
\Omega = (-4,4)x(-4,4) - \Omega_void
\Omega_void = {(x,y):x^2+y^2<1}

@author: cosmin
"""


import jax
import jax.flatten_util
import jax.numpy as jnp
import jaxopt
from jax import config
import time
import matplotlib.pyplot as plt
import numpy as np

from jaxiga.utils.Geom_examples import PlateWHoleQuadrant
from jaxiga.utils_iga.materials import MaterialElast2D
from jaxiga.utils.IGA import IGAMesh2D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming
from jaxiga.utils.Solvers import Elast2D_DEM_Spline
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

#  Neumann B.C. τ(x,y) = [x, y] on the circular boundary
rad_int = 1.0  # interior radius
bound_trac_x = 10.0  # traction at infinity in the x direction


def u_bound_neu(x, y, nx, ny):
    stress = exact_stress(x, y)
    tau_x = nx * stress[0] + ny * stress[2]
    tau_y = nx * stress[2] + ny * stress[1]
    return [tau_x, tau_y]

# The exact solution for error norm computations
def exact_sol(x, y):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    ux = (
        (1 + nu)
        / Emod
        * bound_trac_x
        * (
            1 / (1 + nu) * r * np.cos(th)
            + 2 * rad_int ** 2 / ((1 + nu) * r) * np.cos(th)
            + rad_int ** 2 / (2 * r) * np.cos(3 * th)
            - rad_int ** 4 / (2 * r ** 3) * np.cos(3 * th)
        )
    )
    uy = (
        (1 + nu)
        / Emod
        * bound_trac_x
        * (
            -nu / (1 + nu) * r * np.sin(th)
            - (1 - nu) * rad_int ** 2 / ((1 + nu) * r) * np.sin(th)
            + rad_int ** 2 / (2 * r) * np.sin(3 * th)
            - rad_int ** 4 / (2 * r ** 3) * np.sin(3 * th)
        )
    )
    return ux, uy


def _exact_stress(x, y):
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    sigma_rr = bound_trac_x / 2 * (1 - rad_int ** 2 / r ** 2) + bound_trac_x / 2 * (
        1 - 4 * rad_int ** 2 / r ** 2 + 3 * rad_int ** 4 / r ** 4
    ) * np.cos(2 * th)
    sigma_tt = bound_trac_x / 2 * (1 + rad_int ** 2 / r ** 2) - bound_trac_x / 2 * (
        1 + 3 * rad_int ** 4 / r ** 4
    ) * np.cos(2 * th)
    sigma_rt = (
        -bound_trac_x
        / 2
        * (1 + 2 * rad_int ** 2 / r ** 2 - 3 * rad_int ** 4 / r ** 4)
        * np.sin(2 * th)
    )
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
bound_outer_up_left = boundary2D("Neumann", 0, "up", u_bound_neu)
bound_outer_down_left = boundary2D("Neumann", 1, "up", u_bound_neu)
bound_outer_down_right = boundary2D("Neumann", 2, "up", u_bound_neu)
bound_outer_up_right = boundary2D("Neumann", 3, "up", u_bound_neu)
bound_inner_left = boundary2D("Dirichlet", 0, "down_left", u_bound_dir_symy0)
bound_inner_down = boundary2D("Dirichlet", 1, "down_left", u_bound_dir_symx0)
bound_inner_right = boundary2D("Dirichlet", 2, "down_left", u_bound_dir_symy0)
bound_inner_up = boundary2D("Dirichlet", 3, "down_left", u_bound_dir_symx0)

bound_cond = [
    bound_outer_up_left,
    bound_outer_down_left,
    bound_outer_down_right,
    bound_outer_up_right,
    bound_inner_left,
    bound_inner_down,
    bound_inner_right,
    bound_inner_up,
]


## Step 1: Generate the geometry
len_side = 4.0  # length of the side of the plate being modelled
# Patch 1:
patch1 = PlateWHoleQuadrant(rad_int, len_side, 2)
# Patch 2:
patch2 = PlateWHoleQuadrant(rad_int, len_side, 3)
# Patch 3
patch3 = PlateWHoleQuadrant(rad_int, len_side, 4)
# Patch 4
patch4 = PlateWHoleQuadrant(rad_int, len_side, 1)
patches = [patch1, patch2, patch3, patch4]

# Step 2: Degree elevate and refine the geometry
t = time.time()
for patch in patches:
    patch.degreeElev(deg - 2, deg - 1)
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
meshes =[]
for patch in patches:
    mesh = IGAMesh2D(patch)
    mesh.classify_boundary()
    meshes.append(mesh)
elapsed = time.time() - t
print("Mesh initialization took ", elapsed, " seconds")

vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(meshes)
edge_list = gen_edge_list(patch2vertex)
size_basis = zip_conforming(meshes, vertex2patch, edge_list)

time_proc = time.time()
# Generate the Gauss points
# Evaluate the spline basis using reduced integrtion
gauss_quad_u = gen_gauss_pts(deg-1)# + 1)
gauss_quad_v = gen_gauss_pts(deg-1)# + 1)
gauss_rule = [gauss_quad_u, gauss_quad_v]
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

t1 = time.time()
params = model.get_params(model.opt_state)
print("Training (LBFGS)...")
params = model.get_params(model.opt_state)

solver = jaxopt.LBFGS(model.get_loss_and_grads, value_and_grad=True, 
                      maxiter=3000, tol=1e-14, jit = True)
results, state = solver.run(params, R, dR, jnp.squeeze(local_areas), rhs)

print("Iteration: ", state[0].item(), " loss: ", state[1].item())
_, unflatten_params = jax.flatten_util.ravel_pytree(params)
interior_vals = unflatten_params(results)
t2 = time.time()
print("Time taken (BFGS)", t2-t1, "seconds")

sol0 = jnp.zeros((2*size_basis))
sol0 = sol0.at[bcdof].set(bcval)
sol0 = sol0.at[model.trainable_indx].set(interior_vals[:, 0])
sol0 = np.array(sol0)
print("Processing took ", time.time()-time_proc, "seconds")

t = time.time()
output_filename = "elast_2d_plate_w_hole_DEM"
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
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_quad_v = gen_gauss_pts(deg + 1)
gauss_rule_err = [gauss_quad_u, gauss_quad_v]
t = time.time()
rel_L2_err, rel_H1_err = comp_error_norm_elast(
    meshes, material.Cmat, sol0, exact_sol, exact_stress, gauss_rule_err
)
print("Relative L2-norm error is ", rel_L2_err)
print("Relative energy-norm error is ", rel_H1_err)
elapsed = time.time() - t
print("Computing the error norms took", elapsed, " seconds")