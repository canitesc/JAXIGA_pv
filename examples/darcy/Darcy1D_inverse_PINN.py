#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:11:57 2022

@author: cosmin
"""

import jax
import jaxopt
import jax.flatten_util
import jax.numpy as jnp
import itertools
import time
from functools import partial
from jax.example_libraries import optimizers
from jax.nn import tanh
from jax import random, grad, vmap, jit, value_and_grad
from tqdm import trange
import matplotlib.pyplot as plt
from jax import config

from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils.bfgs import minimize as bfgs_minimize

key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

class Darcy1D:
    def __init__(self, rng_key, layers, activation=tanh):
        key, *keys = random.split(rng_key, len(layers))
        self.a = 1.
        params = [list(map(self.init_layer, keys, layers[:-1], layers[1:])), self.a]
        self.activation = activation 
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
       
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
        for W, b in params[0][:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = self.activation(outputs)
        W, b = params[0][-1]
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
    
    def loss(self, params, inputs, targets, bnd_inputs, bnd_targets, x_data, u_data):
        #preds = self.apply(params, inputs)
        d2udx2_pred = self.d2u_vec(params, inputs)
        int_residual = -params[1]*d2udx2_pred - targets
        bnd_pred = self.apply(params, bnd_inputs)
        data_pred = self.apply(params, x_data)
        bnd_residual = bnd_pred - bnd_targets
        data_residual = data_pred - u_data
        int_loss = jnp.mean(jnp.square(int_residual))
        bnd_loss = jnp.mean(jnp.square(bnd_residual))
        data_loss = jnp.mean(jnp.square(data_residual))
        loss = int_loss + 1e4*bnd_loss + data_loss
        return loss
    
    def get_loss_and_grads(self, params, *args):
        loss, grads = value_and_grad(self.loss)(params, *args)
        return loss, grads
    
    @partial(jit, static_argnums=(0,))    
    def step(self, i, opt_state, *args):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, *args)
        return self.opt_update(i, g, opt_state)
    
    def train(self, *args, n_iter = 1000):
        pbar = trange(n_iter)
        # Main training loop
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, *args)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.loss(params, *args)
                
                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
                
xmin = 0.
xmax = 1.
num_pts = 1001

a = 4
exact_sol = lambda x : jnp.sin(2*jnp.pi*x)
rhs_fun = lambda x : a*4*jnp.pi**2*jnp.sin(2*jnp.pi*x)

# prepare the training data
x = jnp.linspace(xmin, xmax, num_pts)[1:-1]
x = jnp.expand_dims(x, 1)

num_data_pts = 10
x_data = jnp.linspace(xmin, xmax, num_data_pts+2)[1:-1]
x_data = jnp.expand_dims(x_data, 1)
u_data = exact_sol(x_data)

model = Darcy1D(key, [1, 60, 60, 1])
params = model.get_params(model.opt_state)



# # exact_sol = lambda x : 0.1*jnp.sin(8*jnp.pi*x) + jnp.tanh(80*x)
# # rhs_fun = lambda x : 0.1*64*jnp.pi**2*jnp.sin(8*jnp.pi*x) + 12800*jnp.tanh(80*x)*\
# #             1/jnp.cosh(80*x)**2

# # exact_sol = lambda x : jnp.sin(2*jnp.pi*x) + 0.1*jnp.sin(50*jnp.pi*x)
# # rhs_fun = lambda x : 4*jnp.pi**2 * jnp.sin(2*jnp.pi*x) + 250*jnp.pi**2*jnp.sin(50*jnp.pi*x)

f = rhs_fun(x)
x_bnd = jnp.array([xmin, xmax])
x_bnd = jnp.expand_dims(x_bnd, 1)
u_bnd = exact_sol(x_bnd)

model.train(x, f, x_bnd, u_bnd, x_data, u_data, n_iter = 4000)

t_start = time.time()
params = model.get_params(model.opt_state)
loss_func = jax_tfp_function_factory(model,  params, x, f, x_bnd, u_bnd, x_data, u_data)

#with jax.disable_jit():
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
params = unflatten_params(results.position)

x_test = jnp.linspace(xmin, xmax, 2*num_pts)
x_test = jnp.expand_dims(x_test, 1)
u_pred = model.u(params, x_test)
u_exact = exact_sol(x_test)
err = jnp.squeeze(u_exact) - u_pred
plt.plot(x_test, u_pred, label = 'Prediction')
plt.plot(x_test, u_exact, label = 'Ground truth')
plt.legend()
plt.show()
print("L2-error norm: {}".format(jnp.linalg.norm(err)/jnp.linalg.norm(u_exact)))


print("Predicted a is", params[1])

plt.plot(x_test, err, label = 'Error')
plt.legend()
plt.show()

# du_pred = model.du_vec(params, x_test)
# du_exact = k*jnp.pi*jnp.cos(k*jnp.pi*x_test)
# err_deriv = jnp.squeeze(du_exact) - du_pred
# plt.plot(x_test, du_pred, label = 'Prediction')
# plt.plot(x_test, du_exact, label = 'Ground truth')
# plt.legend()
# plt.show()
# rel_err = jnp.linalg.norm(err_deriv)/jnp.linalg.norm(du_exact)
# print("Relative error for first derivative: {}".format(rel_err))
