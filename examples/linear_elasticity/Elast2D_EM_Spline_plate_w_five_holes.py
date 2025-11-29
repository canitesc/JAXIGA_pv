#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve a 2D elasticity problem on a plate with several voids

@author: cosmin
"""


import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import config
import time
import matplotlib.pyplot as plt
import numpy as np

from jaxiga.utils.Geom import Geometry2D
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
                                      get_measurement_stresses,
                                      plot_fields_2D)

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 3
num_refinements = 3
output_filename = "elast2d_plate_w_five_holes_DEM"


# Step 0: Define the material properties
Emod = 1e3
nu = 0.3
material = MaterialElast2D(Emod=Emod, nu=nu, plane_type="stress")

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

def make_box_w_void_8p(side, rad, center):
    # generates 8 patches (two each at north_east, north_west, south_west, south_east)
    # for the box with void geometry
    geomData = dict()
    geomData["degree_u"] = 2
    geomData["degree_v"] = 1
    geomData["ctrlpts_size_u"] = 3
    geomData["ctrlpts_size_v"] = 2
    # Set knot vectors
    geomData["knotvector_u"] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    geomData["knotvector_v"] = [0.0, 0.0, 1.0, 1.0]
    geomData["weights"] = [1., 1., np.cos(np.pi/8), 1., 1., 1.]
    cpts_ne_1 = [[rad*np.cos(np.pi/4), rad*np.sin(np.pi/4), 0.],
                 [side/2, side/2, 0.],
                 [rad*np.cos(np.pi/8), rad*np.sin(np.pi/8), 0.],
                 [side/2, side/4, 0.],
                 [rad, 0., 0.],
                 [side/2, 0., 0.]]    
    
    # The coordinates of the ith control point [xi, yi, zi] with 
    # weight wi and shifted by [tx, ty, tz] are [xi + wi*tx, yi+wi+ty, zi+wi*tz]
    shift_mat = np.outer(np.array(geomData["weights"]), np.array(center))
    cpts_ne_1_shifted = np.array(cpts_ne_1)+shift_mat
    geomData["ctrlpts"] =  cpts_ne_1_shifted.tolist()
    
    ne_1 = Geometry2D(geomData)
    
    cpts_ne_2 =  [[0., rad, 0.],
                  [0., side/2, 0.],
                  [rad*np.cos(3*np.pi/8), rad*np.sin(3*np.pi/8), 0.],
                  [side/4, side/2, 0.],
                  [rad*np.cos(np.pi/4), rad*np.sin(np.pi/4), 0.],
                  [side/2, side/2, 0.]]
    cpts_ne_2_shifted = np.array(cpts_ne_2)+shift_mat
    
    geomData["ctrlpts"] = cpts_ne_2_shifted
    ne_2 = Geometry2D(geomData)
    
    #rotate by pi/2
    theta = -np.pi/2
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    )
    
    patches_box = [ne_1, ne_2]
    
    # rotate the ne_1 and ne_2 patches three times to get the other patches
    for i in range(3):
        cpts_ne_1 = np.array(cpts_ne_1)@rot_mat
        cpts_ne_1_shifted = cpts_ne_1+shift_mat
        cpts_ne_2 = np.array(cpts_ne_2)@rot_mat
        cpts_ne_2_shifted = cpts_ne_2+shift_mat
        geomData["ctrlpts"] = cpts_ne_1_shifted.tolist()
        patches_box.append(Geometry2D(geomData))
        geomData["ctrlpts"] = cpts_ne_2_shifted.tolist()
        patches_box.append(Geometry2D(geomData))
            
    return patches_box

def make_box_w_void_4p(side, rad, center):
    # generates 4 patches (at north, west, south, east)
    # for the box with void geometry
    geomData = dict()
    geomData["degree_u"] = 2
    geomData["degree_v"] = 1
    geomData["ctrlpts_size_u"] = 3
    geomData["ctrlpts_size_v"] = 2
    # Set knot vectors
    geomData["knotvector_u"] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    geomData["knotvector_v"] = [0.0, 0.0, 1.0, 1.0]
    geomData["weights"] = [1., 1., np.sqrt(2)/2, 1., 1., 1.]
    cpts_north = [[-rad*np.cos(np.pi/4), rad*np.sin(np.pi/4), 0.],
                 [-side/2, side/2, 0.], 
                 [0., rad, 0.],
                 [0., side/2, 0.],
                 [rad*np.cos(np.pi/4), rad*np.sin(np.pi/4), 0.],
                 [side/2, side/2, 0.]]    
    
    # The coordinates of the ith control point [xi, yi, zi] with 
    # weight wi and shifted by [tx, ty, tz] are [xi + wi*tx, yi+wi+ty, zi+wi*tz]
    shift_mat = np.outer(np.array(geomData["weights"]), np.array(center))
    cpts_north_shifted = np.array(cpts_north)+shift_mat
    geomData["ctrlpts"] =  cpts_north_shifted.tolist()
    
    north = Geometry2D(geomData)
        
    #rotate by pi/2
    theta = -np.pi/2
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    )
    
    patches_box = [north]
    
    # rotate the north patches three times to get the other patches
    for i in range(3):
        cpts_north = np.array(cpts_north)@rot_mat
        cpts_north_shifted = cpts_north+shift_mat
        geomData["ctrlpts"] = cpts_north_shifted.tolist()
        patches_box.append(Geometry2D(geomData))                    
    return patches_box

sides_8p = [12., 12., 12.]
rads_8p = [3., 0.6, 1.]
centers_8p = [[0., 0., 0.], [12., 0., 0.], [6., 12., 0.]]

sides_4p = [6., 6.]
rads_4p = [0.5, 0.5]
centers_4p = [[-3., 9., 0.], [15., 15., 0.]]

quads = [np.array([[-6., 12.], [-6, 18.], [0., 12.], [0., 18.]]),
         np.array([[12., 6.], [12., 12.], [18., 6.], [18., 12.]])]

bound_up_1 = boundary2D("Neumann", 32, "up", u_bound_neu)
bound_up_2 = boundary2D("Neumann", 18, "up", u_bound_neu)
bound_up_3 = boundary2D("Neumann", 17, "up", u_bound_neu)
bound_up_4 = boundary2D("Neumann", 28, "up", u_bound_neu)

bound_down_1 = boundary2D("Dirichlet", 5, "up", u_bound_dir_symy0)
bound_down_2 = boundary2D("Dirichlet", 6, "up", u_bound_dir_symy0)
bound_down_3 = boundary2D("Dirichlet", 13, "up", u_bound_dir_symy0)
bound_down_4 = boundary2D("Dirichlet", 14, "up", u_bound_dir_symy0)

bound_left_1 = boundary2D("Dirichlet", 4, "up", u_bound_dir_symx0)
bound_left_2 = boundary2D("Dirichlet", 3, "up", u_bound_dir_symx0)
bound_left_3 = boundary2D("Dirichlet", 25, "up", u_bound_dir_symx0)
bound_left_4 = boundary2D("Dirichlet", 32, "left", u_bound_dir_symx0)

bound_cond = [bound_up_1, bound_up_2, bound_up_3, bound_up_4,
             bound_down_1, bound_down_2, bound_down_3, bound_down_4,
             bound_left_1, bound_left_2, bound_left_3, bound_left_4]
    
patch_list = []
for i in range(len(sides_8p)):
    patches_box = make_box_w_void_8p(sides_8p[i], rads_8p[i], centers_8p[i])
    patch_list += patches_box
    
for i in range(len(sides_4p)):
    patches_box = make_box_w_void_4p(sides_4p[i], rads_4p[i], centers_4p[i])
    patch_list += patches_box
    
for quad in quads:
    patch_list.append(Quadrilateral(quad))

# Step 2: Degree elevate and refine the geometry
t = time.time()
for patch in patch_list[:-2]:
    patch.degreeElev(deg - 2, deg - 1)
for patch in patch_list[-2:]:
    patch.degreeElev(deg - 1, deg - 1)
elapsed = time.time() - t
print("Degree elevation took ", elapsed, " seconds")

t = time.time()
for patch in patch_list:
    for i in range(num_refinements):
        patch.refine_knotvectors(True, True)

elapsed = time.time() - t
print("Knot insertion took ", elapsed, " seconds")

_, ax = plt.subplots()
for patch in patch_list:
    patch.plotKntSurf(ax)
plt.show()

t = time.time()
meshes =[]
for patch in patch_list:
    mesh = IGAMesh2D(patch)
    mesh.classify_boundary()
    meshes.append(mesh)
elapsed = time.time() - t
print("Mesh initialization took ", elapsed, " seconds")

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
