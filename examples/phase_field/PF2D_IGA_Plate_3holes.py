#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM with splines basis (i.e. IGA)
Solve a 2D phase field problem 
Adapted from: https://github.com/somdattagoswami/IGAPack-PhaseField/blob/master/2D%20Examples/utils/solver_2nd.m

@author: cosmin
"""
# %config InlineBackend.figure_format = "retina"
import pypardiso
import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import config
import time
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy import sparse

from jaxiga.utils_iga.materials import MaterialElast2D
from jaxiga.utils.IGA import IGAMesh2D
from jaxiga.utils.Geom import Geometry2D
from jaxiga.utils.Geom_examples import Quadrilateral
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming

from jaxiga.utils.boundary import boundary2D, applyBCElast2D  # , get_bcdof_bcval
from jaxiga.utils.preprocessing_splines import get_mesh_elem_range
from jaxiga.utils.processing_splines import (evaluate_spline_basis_fem_2d,
                                      evaluate_field_fem_2d,
                                      evaluate_stiff_rhs_fem_2d,
                                      make_rhs,
                                      pde_form_elast_pf_2d,
                                      pde_form_phi_pf_2d,
                                      field_form_scalar_2d,
                                      field_form_strains_2d)
from jaxiga.utils.phase_field import history_plate_w_3holes, pos_strain_energy
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils_iga.postprocessing import (plot_sol2D_elast,
                                      plot_sol2D,
                                      comp_measurement_values,
                                      get_measurements_vector,
                                      get_measurement_stresses,
                                      plot_fields_2D)

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 3
num_refinements = 6
output_filename = "pf2d_plate_3holes"

# Step 0: Define the material and model parameters
Emod = 20.80*1e3  # Young's modulus
nu = 0.3  # Poisson ratio
B = 1e3  # Parameter for initial history function
l = 0.025*4  # Length parameter which controls the spread of the damage
cenerg = 1.  # Critical energy release for unstable crack (Gc)
material = MaterialElast2D(Emod=Emod, nu=nu, plane_type="stress")

num_load_steps = 50
delta_disp = -5e-3
tol_inner = 1e-5  # tolerance for ending the staggered scheme


# symmetry Dirichlet B.C., u_y = 0 for y=0 and u_x=0 for x=0
def u_bound_dir_fixed(x, y):
    return [0.0, 0.0]


def u_bound_dir_fix_y(x, y):
    return [None, 0.]


def u_bound_dir_vert_disp(x, y, disp=0.):
    return [None, disp]


# # Set the boundary conditions
bound_down_left = boundary2D("Dirichlet", 0, "down_right", u_bound_dir_fix_y)
bound_down_right = boundary2D("Dirichlet", 21, "down_left", u_bound_dir_fixed)
bound_up = boundary2D("Dirichlet", 16, "up_right", u_bound_dir_vert_disp)
bound_cond = [bound_down_left, bound_down_right, bound_up]

# get_x_dofs = lambda x, y : [0., None]
# bound_react = [boundary2D("Dirichlet", 0, 'down', get_x_dofs)]

# Step 1: Generate the geometry


def make_quad_grid(x_grid_pts, y_grid_pts):
    '''
    Make a grid of quadrilateral elements with x coordinates at x_grid_pts and 
    y coordinates at y_grid_pts

    Parameters
    ----------
    x_grid_pts : (list or 1D array)
        the x-coordinates of the quad vertices
    y_grid_pts : (list of 1D array)
        the y-coordinates of the quad vertices

    Returns
    -------
    quads : (list of Geom2D objects)
        a list of quadrilateral patches forming a grid

    '''
    quads = []

    for i in range(len(x_grid_pts) - 1):
        for j in range(len(y_grid_pts) - 1):
            # Define the vertices of the quadrilateral in counter-clockwise order
            quad_points = np.array([
                [x_grid_pts[i], y_grid_pts[j]],
                [x_grid_pts[i], y_grid_pts[j + 1]],
                [x_grid_pts[i + 1], y_grid_pts[j]],
                [x_grid_pts[i + 1], y_grid_pts[j + 1]]
            ])
            quads.append(Quadrilateral(quad_points))
    return quads

# def _make_quarter_box(pt_1, pt_2, center, rad, side):
#     # helper function for make_box_with_void


def make_box_with_void(quad_pts, center, rad):
    '''
    Parameterize the geoemtry of a quadrilateral with a circular void inside 
    using four IGA patches

    Parameters
    ----------
    quad_pts : (list of the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        the coordinates of the outer quadrilateral vertices, given in counter-clockwise
        order
    center : (list of the form [x, y, z])
        the coordinates of the void center 
    rad : scalar
        the radius of the void

    Returns
    -------
    patches : list of Geom2D objects
        the parametrized geometry for the quadrilateral with void
    '''
    geomData = dict()
    geomData["degree_u"] = 2
    geomData["degree_v"] = 1
    geomData["ctrlpts_size_u"] = 3
    geomData["ctrlpts_size_v"] = 2
    # Set knot vectors
    geomData["knotvector_u"] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    geomData["knotvector_v"] = [0.0, 0.0, 1.0, 1.0]
    geomData["weights"] = [1., 1., np.sqrt(2)/2, 1., 1., 1.]
    # translate the quad points to where center = (0,0)
    x, y, z = center
    x1 = quad_pts[0][0] - x
    y1 = quad_pts[0][1] - y
    x2 = quad_pts[1][0] - x
    y2 = quad_pts[1][1] - y
    x3 = quad_pts[2][0] - x
    y3 = quad_pts[2][1] - y
    x4 = quad_pts[3][0] - x
    y4 = quad_pts[3][1] - y

    mid_seg_nord = [(x4+x3)/2, (y4+y3)/2]
    mid_seg_west = [(x1+x4)/2, (y1+y4)/2]
    mid_seg_south = [(x1+x2)/2, (y1+y2)/2]
    mid_seg_east = [(x2+x3)/2, (y2+y3)/2]

    cpts_north = [[-rad*np.cos(np.pi/4), rad*np.sin(np.pi/4), 0.],
                  [x4, y4, 0.],
                  [0., rad, 0.],
                  [mid_seg_nord[0], mid_seg_nord[1], 0.],
                  [rad*np.cos(np.pi/4), rad*np.sin(np.pi/4), 0.],
                  [x3, y3, 0.]]
    cpts_west = [[-rad*np.cos(np.pi/4), -rad*np.sin(np.pi/4), 0.],
                 [x1, y1, 0.],
                 [-rad, 0., 0.],
                 [mid_seg_west[0], mid_seg_west[1], 0.],
                 [-rad*np.cos(np.pi/4), rad*np.sin(np.pi/4), 0.],
                 [x4, y3, 0.]]
    cpts_south = [[rad*np.cos(np.pi/4), -rad*np.sin(np.pi/4), 0.],
                  [x2, y2, 0.],
                  [0., -rad, 0.],
                  [mid_seg_south[0], mid_seg_south[1], 0.],
                  [-rad*np.cos(np.pi/4), -rad*np.sin(np.pi/4), 0.],
                  [x1, y1, 0.]]
    cpts_east = [[rad*np.cos(np.pi/4), rad*np.sin(np.pi/4), 0.],
                 [x3, y3, 0.],
                 [rad, 0., 0.],
                 [mid_seg_east[0], mid_seg_east[1], 0.],
                 [rad*np.cos(np.pi/4), -rad*np.sin(np.pi/4), 0.],
                 [x2, y2, 0.]]

    # The coordinates of the ith control point [xi, yi, zi] with
    # weight wi and shifted by [tx, ty, tz] are [xi + wi*tx, yi+wi+ty, zi+wi*tz]
    shift_mat = np.outer(np.array(geomData["weights"]), np.array(center))
    cpts_north_shifted = np.array(cpts_north) + shift_mat
    cpts_west_shifted = np.array(cpts_west) + shift_mat
    cpts_south_shifted = np.array(cpts_south) + shift_mat
    cpts_east_shifted = np.array(cpts_east) + shift_mat
    geomData["ctrlpts"] = cpts_north_shifted.tolist()
    north = Geometry2D(geomData)
    geomData["ctrlpts"] = cpts_west_shifted.tolist()
    west = Geometry2D(geomData)
    geomData["ctrlpts"] = cpts_south_shifted.tolist()
    south = Geometry2D(geomData)
    geomData["ctrlpts"] = cpts_east_shifted.tolist()
    east = Geometry2D(geomData)
    patches_box = [north, west, south, east]

    return patches_box


x_grid_pts_left = [0., 1., 2.5, 4.]
y_grid_pts_left = [0., 1., 3.75, 5.75, 8.]

x_grid_pts_right = [8., 10., 19., 20.]
y_grid_pts_right = y_grid_pts_left

quads_left = make_quad_grid(x_grid_pts_left, y_grid_pts_left)
quads_right = make_quad_grid(x_grid_pts_right, y_grid_pts_right)

quad_mid = [Quadrilateral(np.array([[4., 0.], [4., 1.], [8., 0.], [8., 1.]]))]


quads_box_1 = [[4., 1.], [8., 1.], [8., 3.75], [4., 3.75]]
center_1 = [6., 2.75, 0.]
rad = 0.25

quads_box_2 = [[4., 3.75], [8., 3.75], [8., 5.75], [4., 5.75]]
center_2 = [6., 4.75, 0.]

quads_box_3 = [[4., 5.75], [8., 5.75], [8., 8.], [4., 8.]]
center_3 = [6., 6.75, 0.]

box_1 = make_box_with_void(quads_box_1, center_1, rad)
box_2 = make_box_with_void(quads_box_2, center_2, rad)
box_3 = make_box_with_void(quads_box_3, center_3, rad)

patches = quads_left + quad_mid + quads_right + box_1 + box_2 + box_3

# Step 2: Degree elevate and refine the geometry
t = time.time()
for patch in patches:
    patch.degreeElev(deg - patch.surf.degree_u, deg - patch.surf.degree_v)
elapsed = time.time() - t
print("Degree elevation took", elapsed, "seconds")

t = time.time()
for patch in patches:
    for i in range(num_refinements):
        patch.refine_knotvectors(True, True)
elapsed = time.time() - t
print("Knot insertion took", elapsed, "seconds")

# print("Plotting the mesh...")
# t = time.time()
# _, ax = plt.subplots()
# for patch in patches:
#     patch.plotKntSurf(ax)
# plt.show()
# elapsed = time.time() - t
# print("Plotting the mesh took", elapsed, "seconds")
t = time.time()
meshes = []
for patch in patches:
    mesh = IGAMesh2D(patch)
    mesh.classify_boundary()
    meshes.append(mesh)
elapsed = time.time() - t
print("Mesh initialization took", elapsed, "seconds")

t = time.time()
vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(meshes)
edge_list = gen_edge_list(patch2vertex)
size_basis = zip_conforming(meshes, vertex2patch, edge_list)
elapse = time.time() - t
print("Mesh connectivity took", elapsed, "seconds")
# # Get the DOFs corresponding to the reaction forces
# bcdof_react , _ = get_bcdof_bcval(mesh_list, bound_react)

# Generate the Gauss points
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_quad_v = gen_gauss_pts(deg + 1)
gauss_rule = [gauss_quad_u, gauss_quad_v]


# History function and phase field initialization.
num_fields = 1
R, dR, local_areas, phys_pts, global_nodes_phi = evaluate_spline_basis_fem_2d(meshes,
                                                                              gauss_rule,
                                                                              num_fields)
# set the phase field to zero on elastic meshes to avoid spurious cracks at the
# fixed nodes
elastic_meshes = list(range(0, 9)) + list(range(13, 25))
elem_range = get_mesh_elem_range(meshes)
elems_zero_out = []
for i in elastic_meshes:
    start_index = elem_range[i][0]
    end_index = elem_range[i][1]
    elems_zero_out.append(list(range(start_index, end_index)))

fenerg = history_plate_w_3holes(phys_pts, B, l, cenerg)
phys_pts_plot = jnp.reshape(phys_pts, (-1, 2))
triang = tri.Triangulation(phys_pts_plot[:, 0], phys_pts_plot[:, 1])
plt.tricontourf(triang, fenerg.flatten())
plt.colorbar()
plt.axis('equal')
plt.title('Fracture energy')
plt.show()

assert False

sol_phi = jnp.zeros(size_basis)
num_fields = 2
_, _, _, _, global_nodes_u = evaluate_spline_basis_fem_2d(meshes, gauss_rule,
                                                          num_fields)
num_fields = 1
phi_vals = evaluate_field_fem_2d(R, dR, num_fields, global_nodes_phi,
                                 field_form_scalar_2d, sol_phi, ())  # .flatten()

react_plt = np.zeros((num_load_steps, 2))
top_disp = 0.
for i_load in range(num_load_steps):
    print('Load step', i_load)

    # Inner iteration
    norm_err_inner = float('inf')
    miter = 0
    while norm_err_inner > tol_inner:

        print('Inner iteration', miter)
        num_fields = 2
        II, JJ, S, local_rhss = evaluate_stiff_rhs_fem_2d(R, dR, local_areas, phys_pts,
                                                          gauss_rule, num_fields,
                                                          global_nodes_u,
                                                          pde_form_elast_pf_2d,
                                                          (material,), (phi_vals,))
        t = time.time()
        stiff_mat = sparse.coo_matrix((S, (II, JJ))).tocsr()
        stiff_mat_u = stiff_mat.copy()
        rhs = np.asarray(make_rhs(global_nodes_u, local_rhss,
                         num_fields, size_basis)).copy()
        stiff, rhs = applyBCElast2D(
            meshes, bound_cond, stiff_mat, rhs, gauss_rule)
        elapsed = time.time() - t
        print("Applying B.C.s took ", elapsed, " seconds")

        # Solve the linear system
        t = time.time()
        # sol_u = spsolve(stiff, rhs)
        sol_u = pypardiso.spsolve(stiff, rhs)
        elapsed = time.time() - t
        print("Linear sparse solver took ", elapsed, " seconds")
        # update the strain energy
        t = time.time()
        strain_vals = evaluate_field_fem_2d(R, dR, num_fields, global_nodes_u,
                                            field_form_strains_2d, sol_u, ())

        fenerg = pos_strain_energy(material, strain_vals, fenerg)
        # zero out the fracture energy in the elastic meshes
        fenerg = fenerg.at[elems_zero_out, :].set(0.)

        if miter % 5 == 0:
            plt.tricontourf(triang, fenerg.flatten())
            plt.colorbar()
            plt.title('Positive strain energy at load step ' +
                      str(i_load) + ' inner step '+str(miter))
            plt.show()
        # solve for phi
        num_fields = 1
        II, JJ, S, local_rhss = evaluate_stiff_rhs_fem_2d(R, dR, local_areas, phys_pts,
                                                          gauss_rule, num_fields,
                                                          global_nodes_phi,
                                                          pde_form_phi_pf_2d,
                                                          (cenerg, l), (fenerg,))

        stiff = sparse.coo_matrix((S, (II, JJ))).tocsr()
        rhs = np.asarray(make_rhs(global_nodes_phi, local_rhss,
                         num_fields, size_basis)).copy()
        # stiff, rhs = applyBCElast2D(meshes, bound_cond, stiff_mat, rhs, gauss_rule)
        elapsed = time.time() - t
        print("Setting up the phase field linear system took ", elapsed, " seconds")

        # Solve the linear system
        t = time.time()
        sol_phi_old = sol_phi.copy()
        # sol_phi = spsolve(stiff, rhs)
        sol_phi = pypardiso.spsolve(stiff, rhs)
        norm_err = jnp.linalg.norm(sol_phi-sol_phi_old)
        print('Norm change sol_phi =', norm_err)
        norm_err_inner = jnp.linalg.norm(
            stiff@sol_phi_old-rhs)/jnp.linalg.norm(rhs)
        print('Norm error inner', norm_err_inner)
        elapsed = time.time() - t
        print("Linear sparse solver took ", elapsed, " seconds")

        phi_vals = evaluate_field_fem_2d(R, dR, num_fields, global_nodes_phi,
                                         field_form_scalar_2d, sol_phi, ())  # .flatten()

        if miter % 5 == 0:
            plt.tricontourf(triang, phi_vals.flatten())
            plt.colorbar()
            plt.title('Phase field at load step ' +
                      str(i_load) + ' inner step '+str(miter))
            plt.show()
        miter += 1

    t = time.time()
    # plot_sol2D_elast(meshes, material.Cmat, sol_u,
    #                  output_filename+'_u_step_'+str(i_load))
    plot_sol2D(meshes, sol_phi, output_filename+'_phi_step_'+str(i_load))

    elapsed = time.time() - t
    print("Plotting to VTK took ", elapsed, " seconds")
    t = time.time()
    # compute the displacements at a set of uniformly spaced points
    num_pts_xi = 20
    num_pts_eta = 20
    num_fields = 2
    meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = comp_measurement_values(num_pts_xi,
                                                                                                num_pts_eta,
                                                                                                meshes,
                                                                                                sol_u,
                                                                                                get_measurements_vector,
                                                                                                num_fields)
    elapsed = time.time() - t
    print("Computing the displacement values at measurement points took ",
          elapsed, " seconds")

    t = time.time()
    # compute the stresses at a set of uniformly spaced points
    num_output_fields = 4
    meas_stress_all, meas_pts_phys_xy_all, vals_stress_min, vals_stress_max = comp_measurement_values(num_pts_xi,
                                                                                                      num_pts_eta,
                                                                                                      meshes,
                                                                                                      sol_u,
                                                                                                      get_measurement_stresses,
                                                                                                      num_output_fields,
                                                                                                      material.Cmat)
    # compute phi at a set of uniformly spaced points
    num_fields = 1
    meas_phi_all, meas_pts_phys_xy_all, vals_phi_min, vals_phi_max = comp_measurement_values(num_pts_xi,
                                                                                             num_pts_eta,
                                                                                             meshes,
                                                                                             sol_phi,
                                                                                             get_measurements_vector,
                                                                                             num_fields)
    elapsed = time.time() - t
    print("Computing the phi values at measurement points took ", elapsed, " seconds")

    elapsed = time.time() - t
    print("Computing the stress values at measurement points took ",
          elapsed, " seconds")

    t = time.time()
    field_title = "Computed solution"
    field_names = ['x-disp', 'y-disp']
    plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes,
                   meas_pts_phys_xy_all, meas_vals_all, vals_disp_min, vals_disp_max)
    elapsed = time.time() - t
    print("Plotting the displacement solution (matplotlib) took ", elapsed, " seconds")

    t = time.time()
    field_title = "Computed solution"
    field_names = ['xx-stress', 'yy-stress', 'xy-stress', 'VM-stress']
    plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes,
                   meas_pts_phys_xy_all, meas_stress_all, vals_stress_min, vals_stress_max)
    elapsed = time.time() - t
    print("Plotting the stress solution (matplotlib) took ", elapsed, " seconds")

    t = time.time()
    field_title = "Computed solution"
    field_names = ['phi']
    plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, meshes,
                   meas_pts_phys_xy_all, meas_phi_all, vals_phi_min, vals_phi_max)
    elapsed = time.time() - t
    print("Plotting the stress solution (matplotlib) took ", elapsed, " seconds")

    # react_force = np.sum(-stiff_mat_u[bcdof_react, :]@sol_u)
    # react_plt[i_load, :] = [top_disp, react_force]
    # print("Displacement:", top_disp, "Reaction force:", react_force)

    top_disp += delta_disp

    def u_bound_dir_var_vert_disp(x, y): return [None, top_disp]
    # # Set the boundary conditions
    bound_up = boundary2D("Dirichlet", 16, "up_right",
                          u_bound_dir_var_vert_disp)
    bound_cond = [bound_down_left, bound_down_right, bound_up]

# plt.plot(react_plt[:,0], react_plt[:, 1])
# plt.title("Load vs. displacement")
# plt.show()
