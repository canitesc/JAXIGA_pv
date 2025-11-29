#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM with splines basis (i.e. IGA)
Solve a 2D phase field problem 
Adapted from: https://github.com/somdattagoswami/IGAPack-PhaseField/blob/master/2D%20Examples/utils/solver_2nd.m

@author: cosmin
"""
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

from jaxiga.utils.Geom_examples import Quadrilateral
from jaxiga.utils_iga.materials import MaterialElast2D

from jaxiga.utils.IGA import IGAMesh2D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming

from jaxiga.utils.boundary import boundary2D, applyBCElast2D, get_bcdof_bcval
from jaxiga.utils.processing_splines import (evaluate_spline_basis_fem_2d,
                                      evaluate_field_fem_2d,
                                      evaluate_stiff_rhs_fem_2d,
                                      make_rhs,
                                      pde_form_elast_pf_2d,
                                      pde_form_phi_pf_2d,
                                      field_form_scalar_2d,
                                      field_form_strains_2d)
from jaxiga.utils.phase_field import history_edge_crack, pos_strain_energy
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils_iga.postprocessing import (plot_sol2D_elast,
                                      plot_sol2D,
                                      comp_measurement_values,
                                      get_measurements_vector,
                                      get_measurement_stresses,
                                      plot_fields_2D)

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 4
num_refinements = 7
output_filename = "pf2d_tension_plate"

# Step 0: Define the material and model parameters
Emod = 210e3 # Young's modulus
nu = 0.3 # Poisson ratio
B = 1e3 # Parameter for initial history function
l = 0.0125 # Length parameter which controls the spread of the damage
cenerg = 2.7 # Critical energy release for unstable crack (Gc)
material = MaterialElast2D(Emod=Emod, nu=nu, plane_type="stress")

num_load_steps = 100
delta_disp = 1e-4
tol_inner = 1e-5 # tolerance for ending the staggered scheme


# symmetry Dirichlet B.C., u_y = 0 for y=0 and u_x=0 for x=0
def u_bound_dir_fixed(x, y):
    return [0.0, 0.0]

def u_bound_dir_vert_disp(x, y, disp=0.):
    return [0.0, disp]


# # Set the boundary conditions
bound_down = boundary2D("Dirichlet", 0, "down", u_bound_dir_fixed)
bound_up = boundary2D("Dirichlet", 0, "up", u_bound_dir_vert_disp)
bound_cond = [bound_down, bound_up]

get_y_dofs = lambda x, y : [None, 0.]
bound_react = [boundary2D("Dirichlet", 0, 'down', get_y_dofs)]
              
# Step 1: Generate the geometry
# Patch 1:
_, ax = plt.subplots()
plate_length = 1.
plate_width  = 1.
vertices = [[0., 0.], [0., plate_width], [plate_length, 0.], [plate_length, plate_width]]
patch1 = Quadrilateral(np.array(vertices))
patch1.plotKntSurf(ax)


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

mesh1.classify_boundary()
mesh_list = [mesh1]
vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(mesh_list)
edge_list = gen_edge_list(patch2vertex)
size_basis = zip_conforming(mesh_list, vertex2patch, edge_list)

# Get the DOFs corresponding to the reaction forces
bcdof_react , _ = get_bcdof_bcval(mesh_list, bound_react)

# Generate the Gauss points        
# Evaluate the spline basis
gauss_quad_u = gen_gauss_pts(deg + 1)
gauss_quad_v = gen_gauss_pts(deg + 1)
gauss_rule = [gauss_quad_u, gauss_quad_v]

# History function and phase field initialization.
num_fields = 1
R, dR, local_areas, phys_pts, global_nodes_phi = evaluate_spline_basis_fem_2d(mesh_list,
                                                    gauss_rule,
                                                    num_fields)

fenerg = history_edge_crack(phys_pts, B, l, cenerg)
phys_pts_plt = jnp.reshape(phys_pts, (-1, 2))
triang = tri.Triangulation(phys_pts_plt[:,0], phys_pts_plt[:,1])
plt.tricontourf(triang, fenerg.flatten())
plt.colorbar()
plt.title('Fracture energy')
plt.show()

num_fields = 2
_, _, _, _, global_nodes_u = evaluate_spline_basis_fem_2d(mesh_list, gauss_rule,
                                                          num_fields)

sol_phi = jnp.zeros(size_basis)
num_fields = 1

phi_vals = evaluate_field_fem_2d(R, dR, num_fields, global_nodes_phi,
                                   field_form_scalar_2d, sol_phi, ())#.flatten()


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
        rhs = np.asarray(make_rhs(global_nodes_u, local_rhss, num_fields, size_basis)).copy()
        stiff, rhs = applyBCElast2D(mesh_list, bound_cond, stiff_mat, rhs, gauss_rule)
        elapsed = time.time() - t
        print("Applying B.C.s took ", elapsed, " seconds")
        
        # Solve the linear system
        t = time.time()
        sol_u = pypardiso.spsolve(stiff, rhs)
        elapsed = time.time() - t
        print("Linear sparse solver took ", elapsed, " seconds")
        t = time.time()

        # update the strain energy
        strain_vals = evaluate_field_fem_2d(R, dR, num_fields, global_nodes_u,
                                            field_form_strains_2d, sol_u, ())
               
        fenerg = pos_strain_energy(material, strain_vals, fenerg)
        if miter % 5 == 0:
            plt.tricontourf(triang, fenerg.flatten())
            plt.colorbar()
            plt.title('Positive strain energy at load step ' + str(i_load) +' inner step '+str(miter))
            plt.show()
        # solve for phi
        num_fields = 1
        II, JJ, S, local_rhss = evaluate_stiff_rhs_fem_2d(R, dR, local_areas, phys_pts,
                                                          gauss_rule, num_fields,
                                                          global_nodes_phi,
                                                          pde_form_phi_pf_2d,
                                                          (cenerg, l), (fenerg,))
        stiff = sparse.coo_matrix((S, (II, JJ))).tocsr()
        rhs = np.asarray(make_rhs(global_nodes_phi, local_rhss, num_fields, size_basis)).copy()
        #stiff, rhs = applyBCElast2D(mesh_list, bound_cond, stiff_mat, rhs, gauss_rule)
        elapsed = time.time() - t
        print("Setting up the phase field linear system took ", elapsed, " seconds")
        
        # Solve the linear system
        t = time.time()
        sol_phi_old = sol_phi.copy()
        sol_phi = pypardiso.spsolve(stiff, rhs)
        norm_err = jnp.linalg.norm(sol_phi-sol_phi_old)
        print('Norm change sol_phi =', norm_err)
        norm_err_inner = jnp.linalg.norm(stiff@sol_phi_old-rhs)/jnp.linalg.norm(rhs)
        print('Norm error inner', norm_err_inner)
        miter += 1
        elapsed = time.time() - t
        print("Linear sparse solver took ", elapsed, " seconds")                
        
        phi_vals = evaluate_field_fem_2d(R, dR, num_fields, global_nodes_phi, 
                                         field_form_scalar_2d, sol_phi, ())
        
    t = time.time()
    plot_sol2D_elast(mesh_list, material.Cmat, sol_u, output_filename+'_u_step_'+str(i_load))
    plot_sol2D(mesh_list, sol_phi, output_filename+'_phi_step_'+str(i_load))
    
    elapsed = time.time() - t
    print("Plotting to VTK took ", elapsed, " seconds")
    t = time.time()
    # compute the displacements at a set of uniformly spaced points
    num_pts_xi = 100
    num_pts_eta = 100
    num_fields = 2
    meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = comp_measurement_values(num_pts_xi,
                                                                  num_pts_eta,
                                                                  mesh_list,
                                                                  sol_u,
                                                                  get_measurements_vector,
                                                                  num_fields)
    elapsed = time.time() - t
    print("Computing the displacement values at measurement points took ", elapsed, " seconds")
    
    t = time.time()
    # compute the stresses at a set of uniformly spaced points
    num_output_fields = 4
    meas_stress_all, meas_pts_phys_xy_all, vals_stress_min, vals_stress_max = comp_measurement_values(num_pts_xi,
                                                                  num_pts_eta,
                                                                  mesh_list,
                                                                  sol_u,
                                                                  get_measurement_stresses,
                                                                  num_output_fields,
                                                                  material.Cmat)
    # compute phi at a set of uniformly spaced points
    num_fields = 1
    meas_phi_all, meas_pts_phys_xy_all, vals_phi_min, vals_phi_max = comp_measurement_values(num_pts_xi,
                                                                  num_pts_eta,
                                                                  mesh_list,
                                                                  sol_phi,
                                                                  get_measurements_vector,
                                                                  num_fields)
    elapsed = time.time() - t
    print("Computing the phi values at measurement points took ", elapsed, " seconds")
    
    
    elapsed = time.time() - t
    print("Computing the stress values at measurement points took ", elapsed, " seconds")
    
    
    t = time.time()
    field_title = "Computed solution"
    field_names = ['x-disp', 'y-disp']
    plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list, 
                    meas_pts_phys_xy_all, meas_vals_all, vals_disp_min, vals_disp_max)
    elapsed = time.time() - t
    print("Plotting the displacement solution (matplotlib) took ", elapsed, " seconds")
    
    t = time.time()
    field_title = "Computed solution"
    field_names = ['xx-stress', 'yy-stress', 'xy-stress', 'VM-stress']
    plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list, 
                    meas_pts_phys_xy_all, meas_stress_all, vals_stress_min, vals_stress_max)
    elapsed = time.time() - t
    print("Plotting the stress solution (matplotlib) took ", elapsed, " seconds")
    
    t = time.time()
    field_title = "Computed solution"
    field_names = ['phi']
    plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list, 
                    meas_pts_phys_xy_all, meas_phi_all, vals_phi_min, vals_phi_max)
    elapsed = time.time() - t
    print("Plotting the stress solution (matplotlib) took ", elapsed, " seconds")
            
    
    react_force = np.sum(-stiff_mat_u[bcdof_react, :]@sol_u)
    react_plt[i_load, :] = [top_disp, react_force]
    print("Displacement:", top_disp, "Reaction force:", react_force)
    
    
    top_disp += delta_disp
    u_bound_dir_var_vert_disp = lambda x, y : [0.0, top_disp]
    # # Set the boundary conditions    
    bound_up = boundary2D("Dirichlet", 0, "up", u_bound_dir_var_vert_disp)
    bound_cond = [bound_down, bound_up]

plt.plot(react_plt[:,0], react_plt[:, 1])
plt.title("Load vs. displacement")
plt.show()
