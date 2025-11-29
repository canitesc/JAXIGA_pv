#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimize \int_\Omega (d^2 + l0^2.2*d'(x)^2+l0^4/16*d''(x)^2) dx
corresponding to the energy functional in the fourth order phase field method
                      
with Deep Energy Method
@author: cosmin
"""

import jax
import jax.flatten_util
import jax.numpy as jnp
import itertools
import time
from functools import partial
from jax.example_libraries import optimizers
from jax.nn import swish
from jax import random, grad, vmap, jit, value_and_grad
from tqdm import trange
import matplotlib.pyplot as plt
from jax import config

from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils.preprocessing_DEM_1d import generate_quad_pts_weights_1d
from jaxiga.utils.bfgs import minimize as bfgs_minimize
from jaxiga.utils.postprocessing import plot_convergence_dem


key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

class PF1D_4th_DEM:
    def __init__(self, rng_key, l0, layers, activation=swish):
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(self.init_layer, keys, layers[:-1], layers[1:]))
        self.activation = activation 
        self.l0 = l0
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-4)
       
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []

    def init_layer(self, key, d_in, d_out):
        k1, k2 = random.split(key)
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b
    
    def apply(self, params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = self.activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs
    
    def u(self, params, x):
        u_val = jnp.squeeze(self.apply(params, x))
        return u_val
    
    def du(self, params, x):
        dudx = jnp.squeeze(grad(self.u, 1)(params, x))
        return dudx
    
    def du_vec(self, params, inputs):
        dudx = vmap(self.du, (None, 0))(params, inputs)
        return dudx                
    
    def d2u(self, params, x):
        d2udx2 = grad(self.du, 1)(params, x)
        return d2udx2
    
    def d2u_vec(self, params, inputs):
        d2udx2 = vmap(self.d2u, (None, 0))(params, inputs)
        return d2udx2
    
    def all_losses(self, params, coords, rhs, weights, bnd_inputs, bnd_targets):
        #preds = self.apply(params, inputs)
        u_pred = self.u(params, coords)
        dudx_pred = self.du_vec(params, coords)
        d2udx2_pred = self.d2u_vec(params, coords)
        # int_energy = (self.l0**4/16*d2udx2_pred**2+self.l0**2/2*dudx_pred**2 +\
        #               u_pred**2)*jnp.squeeze(weights)
        int_energy = (self.l0**4/2**d2udx2_pred**2 + self.l0**2*dudx_pred**2 + \
                      u_pred**2)*jnp.squeeze(weights)
        #int_energy = (self.l0**4*d2udx2_pred**2 + 2*self.l0**2*dudx_pred**2 + \
        #               u_pred**2)*jnp.squeeze(weights)
        bnd_pred = self.apply(params, bnd_inputs)
        bnd_residual = bnd_pred - bnd_targets
        int_loss = jnp.sum(int_energy)
        bnd_loss = jnp.mean(jnp.square(bnd_residual))
        return int_loss, bnd_loss
    
    def loss(self, params, coords, rhs, weights, bnd_inputs, bnd_targets):
        int_loss, bnd_loss = self.all_losses(params, coords, rhs, weights, bnd_inputs, bnd_targets)
        loss = int_loss + 1e6*bnd_loss
        return loss
    
    def get_loss_and_grads(self, params, *args):
        loss, grads = value_and_grad(self.loss)(params, *args)
        return loss, grads
    
    @partial(jit, static_argnums=(0,))    
    def step(self, i, opt_state, coords, rhs, weights, bnd_inputs, bnd_targets):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, coords, rhs, weights, bnd_inputs, bnd_targets)
        return self.opt_update(i, g, opt_state)
    
    def train(self, coords, rhs, weights, bnd_inputs, bnd_targets, n_iter = 1000):
        pbar = trange(n_iter)
        # Main training loop
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       coords, rhs, weights, bnd_inputs, bnd_targets)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                int_loss, bnd_loss = self.all_losses(params, coords, rhs, weights, bnd_inputs,
                                       bnd_targets)
                loss_value = int_loss + bnd_loss
                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss': loss_value, 'Bnd_loss': bnd_loss})


# define the problem
l0 = 0.25
exact_sol = lambda x : jnp.exp(-jnp.abs(2*x)/l0)*(1+jnp.abs(2*x)/l0)
rhs_fun = lambda x : jnp.zeros_like(x)
           
#define the input and output data set
x_min = -2.
x_max = 2.
num_elem = 999
num_gauss_pts = 8
data_type = "float64"

Xint, Wint = generate_quad_pts_weights_1d(x_min, x_max, num_elem, num_gauss_pts)
Xint = jnp.array(Xint)[jnp.newaxis].T.astype(data_type)
Yint = rhs_fun(Xint)
Wint = jnp.array(Wint)[jnp.newaxis].T.astype(data_type)


model = PF1D_4th_DEM(key, l0, [1, 16, 16, 1])
params = model.get_params(model.opt_state)


rhs = rhs_fun(Xint)
x_bnd = jnp.array([x_min, 0. , x_max])
x_bnd = jnp.expand_dims(x_bnd, 1)
u_bnd = exact_sol(x_bnd)

#loss = model.loss(params, x, f, x_bnd, u_bnd)
model.train(Xint, Yint, Wint, x_bnd, u_bnd, n_iter = 17500)
params = model.get_params(model.opt_state)

# t_start = time.time()
# params = model.get_params(model.opt_state)
# loss_func = jax_tfp_function_factory(model, params, Xint, Yint, Wint, x_bnd, u_bnd)

# #with jax.disable_jit():
# initial_pos = loss_func.init_params_1d
# num_bfgs_iterations = 0
# results = bfgs_minimize(loss_func, initial_position = initial_pos,
#                     max_iterations=1000)
# new_pos = results.position
# current_loss, _ = loss_func(new_pos)
# num_bfgs_iterations += results.num_iterations
# print("Iteration: ", num_bfgs_iterations, " loss: ", current_loss)
# print("Time taken (BFGS) is ", time.time() - t_start)

# _, unflatten_params = jax.flatten_util.ravel_pytree(params)
# params = unflatten_params(results.position)


x_test = jnp.linspace(x_min, x_max, 2*len(Xint))
x_test = jnp.expand_dims(x_test, 1)
u_pred = model.u(params, x_test)
u_exact = exact_sol(x_test)
err = jnp.squeeze(u_exact) - u_pred
plt.plot(x_test, u_pred, label = 'Prediction')
plt.plot(x_test, u_exact, label = 'Ground truth')
plt.title('u(x)')
plt.legend()
plt.show()
print("L2-error norm: {}".format(jnp.linalg.norm(err)/jnp.linalg.norm(u_exact)))


plt.plot(err)
plt.title("Prediction error")
plt.show()


#plot_convergence_dem(model.loss_log, loss_func.history)
plt.plot(model.loss_log)
plt.title('Loss convergence')
plt.show()






