#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics informed neural network based on colocation with splines basis

@author: cosmin
"""
#%%
#%config InlineBackend.figure_format = "retina"

import jax
import jax.flatten_util
import jax.numpy as jnp
from jax import config
import time

from jaxiga.utils.Solvers import Poisson1D_PINN_Spline
from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils.bfgs import minimize as bfgs_minimize

from jaxiga.utils.bernstein import bezier_extraction
from jaxiga.utils.processing_splines import (eval_spline_basis, 
                                      eval_spline_basis_deriv,
                                      eval_spline_basis_2nd_deriv)
from jaxiga.utils.preprocessing_splines import (make_IEN, make_Greville, 
                                         make_point_to_element_connectivity,
                                         make_uknots,
                                         map_parameter_to_reference)
from jaxiga.utils.postprocessing_splines import (plot_solution_or_deriv,
                                          plot_error_solution_or_deriv)
key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

# Training data for  PINN
N = 200 # number of knot-spans
deg = 6 # polynomial degree
bnd_left_val = 0.
bnd_right_val = 0.

# generate the knot-vector
x_min = 0.
x_max = 1.
dim_space = N+deg
nodes = jnp.linspace(x_min, x_max, N+1)
knots = jnp.concatenate((jnp.zeros(deg), nodes, jnp.ones(deg)))
C, nb = bezier_extraction(knots, deg)

# calculate the spline basis
# 1. IEN array
IEN = make_IEN(knots, deg)
# 2. generate Greville abscissae
G = make_Greville(knots, deg)

# 3a. Make uknots
uknots = make_uknots(knots)
assert (uknots == jnp.unique(knots)).all()

# 3b. point to knot-span connectivity
p2e = make_point_to_element_connectivity(G, uknots)

# 4. map the Greville abscissae to the reference element
xhat = map_parameter_to_reference(G, uknots, p2e)

# 5. evaluate the spline basis
ddNval = eval_spline_basis_2nd_deriv(xhat, uknots, p2e, C, deg)

k = 8
exact_sol = lambda x : jnp.sin(k*jnp.pi*x)
rhs_fun = lambda x : k**2*jnp.pi**2*jnp.sin(k*jnp.pi*x)

fvals = rhs_fun(G)

model = Poisson1D_PINN_Spline(IEN, p2e[1:-1], deg, dim_space)
model.train(ddNval[1:-1], fvals[1:-1])

t_start = time.time()
params = model.get_params(model.opt_state)
loss_func = jax_tfp_function_factory(model,  params, ddNval[1:-1], fvals[1:-1])
initial_pos = loss_func.init_params_1d
tolerance = 1e-5
current_loss, _ = loss_func(initial_pos)
num_bfgs_iterations = 0
while current_loss > tolerance:
    results = bfgs_minimize(loss_func, initial_position = initial_pos,
                        max_iterations=1000)
    initial_pos = results.position
    num_bfgs_iterations += results.num_iterations
    if current_loss < results.objective_value-tolerance:
        current_loss = results.objective_value
    else:
        break
    print("Iteration: ", num_bfgs_iterations, " loss: ", current_loss)
print("Time taken (BFGS) is ", time.time() - t_start)

_, unflatten_params = jax.flatten_util.ravel_pytree(params)
sol0 = unflatten_params(results.position)

#sol0 = model.get_params(model.opt_state)
sol0 = jnp.concatenate((jnp.array([[0.]]), sol0,  jnp.array([[0.]])), axis=0)


num_plot_pts = 101
x_plot = jnp.linspace(0, 1, num_plot_pts)
p2e_plot = make_point_to_element_connectivity(x_plot, uknots).reshape(-1,1)
uhat_plot = map_parameter_to_reference(x_plot, uknots, p2e_plot)
N_plot = eval_spline_basis(uhat_plot, p2e_plot, C, deg)
dN_plot = eval_spline_basis_deriv(uhat_plot, uknots, p2e_plot, C, deg)
ddN_plot = eval_spline_basis_2nd_deriv(uhat_plot, uknots, p2e_plot, C, deg)

plot_title = "Solution"
plot_solution_or_deriv(sol0, x_plot, p2e_plot, uhat_plot, N_plot, IEN, deg, plot_title)

plot_title = "Solution Error"
plot_error_solution_or_deriv(sol0, x_plot, p2e_plot, uhat_plot, N_plot, IEN, deg,
                                  exact_sol, plot_title)
