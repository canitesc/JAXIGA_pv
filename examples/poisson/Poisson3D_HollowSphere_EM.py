#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM with splines basis (i.e. IGA)

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
from scipy import sparse
from scipy.sparse.linalg import spsolve

from jaxiga.utils.Geom_examples import HollowSphere
from jaxiga.utils.preprocessing_splines import get_bcdof
from jaxiga.utils.IGA import IGAMesh3D
from jaxiga.utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming
from jaxiga.utils.Solvers import Poisson3D_DEM_Spline
from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils.bfgs import minimize as bfgs_minimize
from jaxiga.utils.boundary3D import boundary3D, applyBC3D
from jaxiga.utils.processing_splines_3d import (evaluate_spline_basis_fem_3d, 
                                      make_rhs,
                                      pde_form_poisson_3d)
from jaxiga.utils_iga.assembly import gen_gauss_pts
from jaxiga.utils_iga.postprocessing import (plot_sol3D,
                                      plot_sol3D_error,
                                      gen_param_weights_3d,
                                      get_physpts_3d,
                                      comp_error_norm_3d)

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

deg = 2
num_refinements = 2
rad_int = 1.
rad_ext = 4.


# Step 0: Define the function and boundary conditions
def a0(x, y, z):
    return 1.0

def f(x, y, z):
    return -2*x*y*z * (22*x**2 + 22*y**2 + 22*z**2 - 9*(rad_int**2 + rad_ext**2))


def u_bound(x, y, z):
    return 0.0

# The exact solution for error norm computations
def exact_sol(x, y, z):
    return x*y*z*(x**2 + y**2 + z**2 - rad_int**2)*(x**2 + y**2 + z**2 - rad_ext**2)

def deriv_exact_sol(x, y, z):
    return [
        y*z*(2*x**2*(-rad_ext**2 + x**2 + y**2 + z**2) + 2*x**2*(-rad_int**2 + x**2 + y**2 + z**2) + \
             (-rad_ext**2 + x**2 + y**2 + z**2)*(-rad_int**2 + x**2 + y**2 + z**2)),
        x*z*(2*y**2*(-rad_ext**2 + x**2 + y**2 + z**2) + 2*y**2*(-rad_int**2 + x**2 + y**2 + z**2) + \
             (-rad_ext**2 + x**2 + y**2 + z**2)*(-rad_int**2 + x**2 + y**2 + z**2)),
        x*y*(2*z**2*(-rad_ext**2 + x**2 + y**2 + z**2) + 2*z**2*(-rad_int**2 + x**2 + y**2 + z**2) + \
             (-rad_ext**2 + x**2 + y**2 + z**2)*(-rad_int**2 + x**2 + y**2 + z**2))
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
local_areas = jnp.reshape(local_areas, (-1, 1))

# Get the boundary degrees of freedom and set them to zeros
bcdof = get_bcdof(bound_cond, meshes)
bcval = jnp.zeros(len(bcdof))

model = Poisson3D_DEM_Spline(global_nodes_all, deg,
                              size_basis, bcdof, bcval)

phys_pts_all = jnp.reshape(phys_pts, (-1, 3))
fvals = f(phys_pts_all[:, [0]], phys_pts_all[:, [1]], phys_pts_all[:, [2]])

params = model.get_params(model.opt_state)
t_start = time.time()

loss_func = jax_tfp_function_factory(model,  params, R, dR, fvals, local_areas)
initial_pos = loss_func.init_params_1d
tolerance = 1e-6
current_loss, _ = loss_func(initial_pos)
print("Initial loss is ", current_loss)
solver = jaxopt.LBFGS(model.get_loss_and_grads, value_and_grad=True,
                      maxiter=500, tol=1e-14, jit = True)
results, state = solver.run(params, R, dR, fvals, local_areas)

print("Iteration: ", state[0].item(), " loss: ", state[1].item())
_, unflatten_params = jax.flatten_util.ravel_pytree(params)  
    
print("Time taken (BFGS) is ", time.time() - t_start)
interior_vals = unflatten_params(results)

sol0 = jnp.zeros((size_basis))
sol0 = sol0.at[bcdof].set(bcval)
sol0 = sol0.at[model.trainable_indx].set(interior_vals[:,0])

time_end_process = time.time()
print("Total time for processing ", time_end_process - time_start_process)

t = time.time()
output_filename = "poisson_3d_HollowSphere_DEM"
plot_sol3D(meshes, sol0, output_filename)
plot_sol3D_error(meshes, sol0, exact_sol, output_filename+"_err")
elapsed = time.time() - t
print("Plotting to VTK took ", elapsed, " seconds")


# Compute the norm of the error
# FIX: Why is this much slower than the FEM code?
t = time.time()
rel_L2_err, rel_H1_err = comp_error_norm_3d(meshes, sol0, exact_sol,
                                          deriv_exact_sol, a0,
                                          gauss_rule
)
print("Relative L2-norm error is ", rel_L2_err)
print("Relative energy-norm error is ", rel_H1_err)
elapsed = time.time() - t
print("Computing the error norms took", elapsed, " seconds")
