#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM with splines basis (i.e. IGA)

Poisson 3D Hollow Sphere example

@author: cosmin
"""
import jax
import jaxopt
import jax.flatten_util
import jax.numpy as jnp
from jax import config
import time
import matplotlib.pyplot as plt
import numpy as np

from jaxiga.utils.Geom_examples import HollowSphere
from jaxiga.utils.IGA import IGAMesh3D
from jaxiga.utils.Solvers import Elast3D_DEM_Spline

from jaxiga.utils.boundary3D import boundary3D, applyBCElast_DEM_3D, get_bcdof_bcval
from jaxiga.utils.processing_splines_3d import evaluate_spline_basis_fem_3d
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils_iga.materials import MaterialElast3D
from jaxiga.utils_iga.postprocessing import (plot_sol3D_elast,
                                      gen_param_weights_3d,
                                      get_physpts_3d,
                                      cart2sph,
                                      sph2cart,
                                      comp_error_norm_elast_3d)

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 2
num_refinements = 4

pressure = 10.
rad_int = 1.
rad_ext = 4.
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
    return [-nx*pressure, -ny*pressure, -nz*pressure]

# The exact solution for error norm computations
def exact_sol(x, y, z):
    # Exact displacement for the hollow sphere problem
  

    # Convert Cartesian coordinates to spherical coordinates
    azimuth, elevation, r = cart2sph(x, y, z)
    u_r = pressure * rad_int**3 * r / (Emod * (rad_ext**3 - rad_int**3)) * ((1 - \
                                     2 * nu) + (1 + nu) * rad_ext**3 / (2 * r**3))

    ux, uy, uz = sph2cart(azimuth, elevation, u_r)
    

    return ux, uy, uz

def exact_stress(x, y, z):
    # Convert Cartesian coordinates to spherical coordinates
    azimuth, elevation, r = cart2sph(x, y, z)
    phi = azimuth
    theta = np.pi / 2 - elevation
    sigma_r = pressure * rad_int**3 * (rad_ext**3 - r**3) / (r**3 * (rad_int**3 -\
                                                                     rad_ext**3))
    sigma_th = pressure * rad_int**3 * (rad_ext**3 + 2 * r**3) / (2 * r**3 * \
                                                    (rad_ext**3 - rad_int**3))

    rot_mat = np.array([
        [np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), -np.sin(phi)],
        [np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.cos(phi)],
        [np.cos(theta), -np.sin(theta), 0]
    ])

    stress_cart = rot_mat @ np.diag([sigma_r, sigma_th, sigma_th]) @ rot_mat.T

    sigma = np.zeros(6)
    sigma[0] = stress_cart[0, 0]
    sigma[1] = stress_cart[1, 1]
    sigma[2] = stress_cart[2, 2]
    sigma[3] = stress_cart[0, 1]
    sigma[4] = stress_cart[1, 2]
    sigma[5] = stress_cart[0, 2]
    return sigma
    

bound_left = boundary3D("Dirichlet", 0, "left", u_bound_dir_symz0)
bound_back = boundary3D("Dirichlet", 0, "back", u_bound_dir_symx0)
bound_front = boundary3D("Dirichlet", 0, "front", u_bound_dir_symy0)
bound_up = boundary3D("Neumann", 0, "up", u_bound_neu)
bound_cond = [bound_left, bound_back, bound_front, bound_up]

# Generate the geometry
# Patch 1:

patch1 = HollowSphere(rad_int, rad_ext)

coefs = patch1.getCoefs()
patch1.setCoefs(coefs)

# Step 2: Degree elevate and refine the geometry
t = time.time()
patch1.degreeElev(deg - 2, deg - 2, deg-1)
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
gauss_quad_u = gen_gauss_pts(deg)
gauss_quad_v = gen_gauss_pts(deg)
gauss_quad_w = gen_gauss_pts(deg)
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
local_areas = jnp.reshape(local_areas, (-1, 1))

# Get the boundary degrees of freedom and set them to zeros
bcdof, bcval = get_bcdof_bcval(meshes, bound_cond)

model = Elast3D_DEM_Spline(global_nodes_all, deg,
                              size_basis, bcdof, bcval, material)

phys_pts_all = jnp.reshape(phys_pts, (-1, 3))
rhs = applyBCElast_DEM_3D(meshes, bound_cond, size_basis, gauss_rule, model.index_map)

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

sol0 = jnp.zeros((3*size_basis))
sol0 = sol0.at[bcdof].set(bcval)
sol0 = sol0.at[model.trainable_indx].set(interior_vals[:, 0])
sol0 = np.array(sol0)

time_end_process = time.time()
print("Total time for processing ", time_end_process - time_start_process)
t = time.time()
output_filename = "elast_3d_hemisphere_DEM"
plot_sol3D_elast(meshes, material.Cmat, sol0, output_filename)
#plot_sol3D_error(meshes, sol0, exact_sol, output_filename+"_err")
elapsed = time.time() - t
print("Plotting to VTK took ", elapsed, " seconds")

# Compute the norm of the error
t = time.time()
rel_L2_err, rel_H1_err = comp_error_norm_elast_3d(meshes, material.Cmat, sol0,
                                                  exact_sol, exact_stress, gauss_rule)
print("Relative L2-norm error is ", rel_L2_err)
print("Relative energy-norm error is ", rel_H1_err)
elapsed = time.time() - t
print("Computing the error norms took", elapsed, " seconds")
