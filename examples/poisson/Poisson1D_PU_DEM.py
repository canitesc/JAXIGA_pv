#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve Poisson1D equation using PU_DEM method

@author: cosmin
"""
# %%

import jax
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
#from tensorflow_probability.substrates.jax.optimizer import bfgs_minimize

from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils.preprocessing_DEM_1d import generate_quad_pts_weights_1d
from jaxiga.utils.bfgs import minimize as bfgs_minimize


key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

class Poisson1D_PU_DEM:
    def __init__(self, key, layers, nodes, u_bnd, activation=tanh):
        params = []
        for layer in layers:
            key, *keys = random.split(key, len(layer))
            params.append(list(map(self.init_layer, keys, layer[:-1], layer[1:])))
        self.activation = activation 
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
       
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []
        self.nodes = nodes
        self.num_subdomains = len(sub_dom_nodes)-1
        self.pu_coefs = jnp.concatenate((u_bnd[0], jnp.array((self.num_subdomains-1)*[1.]),
                                          u_bnd[1]))
        
    def init_layer(self, key, d_in, d_out):
        k1, k2 = random.split(key)
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b
    
    def apply(self, param, x):
        for W, b in param[:-1]:
            output = jnp.dot(x, W) + b
            x = self.activation(output)
        W, b = param[-1]
        output = jnp.dot(x, W) + b
        return output   
    
    def u(self, param_left, param_right, x, indx):
        if param_left is not None:
            u_net_left = jnp.squeeze(self.apply(param_left, x))
        else:
            u_net_left = 1.
        if param_right is not None:
            u_net_right = jnp.squeeze(self.apply(param_right, x))
        else:
            u_net_right = 1.
        phi_left = (self.nodes[indx+1]-x)/(self.nodes[indx+1]-self.nodes[indx])
        phi_right = (x-self.nodes[indx])/(self.nodes[indx+1]-self.nodes[indx])
        u_val = self.pu_coefs[indx]*phi_left*u_net_left + \
            self.pu_coefs[indx+1]*phi_right*u_net_right        
        return jnp.squeeze(u_val)
    
    def u_vec(self, param_left, param_right, x, indx):
        u_val = vmap(self.u, (None, None, 0, None))(param_left, param_right, x, indx)
        return u_val
    
    def u_all(self, params, xs):
        u_vals = []
        for indx in range(self.num_subdomains):
            if indx==0:
                params_left = None
                params_right = params[indx]
            elif indx==self.num_subdomains-1:
                params_left = params[indx-1]
                params_right = None
            else:
                params_left = params[indx-1]
                params_right = params[indx]
            u_val = self.u_vec(params_left, params_right, xs[indx], indx)
            u_vals.append(u_val)
        return u_vals
    
    def du(self, param_left, param_right, x, indx):  
        dudx = jnp.sum(grad(self.u, 2)(param_left, param_right, x, indx))
        return dudx
        
    def du_vec(self, param_left, param_right, x, indx):
        dudx = vmap(self.du, (None, None, 0, None))(param_left, param_right, x, indx)
        return dudx
    
    def du_all(self, params, xs):
        dudx_vals = []
        for indx in range(self.num_subdomains):
            if indx==0:
                params_left = None
                params_right = params[indx]
            elif indx==self.num_subdomains-1:
                params_left = params[indx-1]
                params_right = None
            else:
                params_left = params[indx-1]
                params_right = params[indx]
            dudx_val = self.du_vec(params_left, params_right, xs[indx], indx)
            dudx_vals.append(dudx_val)
        return dudx_vals                   

    def loss(self, params, coords, rhs, weights):
        int_energy = 0.        
        dudx_pred = self.du_all(params, coords)
        u_pred = self.u_all(params, coords)
        for i in range(self.num_subdomains):
            vec_energy = (1/2*dudx_pred[i]**2-u_pred[i]*jnp.squeeze(rhs[i]))*jnp.squeeze(weights[i])
            int_energy += jnp.sum(vec_energy)
        return int_energy
    
    def get_loss_and_grads(self, params, *args):
        loss, grads = value_and_grad(self.loss)(params, *args)
        return loss, grads
    
    @partial(jit, static_argnums=(0,))    
    def step(self, i, opt_state, coords, rhs, weights):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, coords, rhs, weights)
        return self.opt_update(i, g, opt_state)
    
    def train(self, coords, rhs, weights, n_iter = 1000):
        pbar = trange(n_iter)
        # Main training loop
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       coords, rhs, weights)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.loss(params, coords, rhs, weights)
                
                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
                
xmin = -1.
xmax = 1.

sub_dom_nodes = [-1., -1/3, 1/3, 1.]
num_subdomains = len(sub_dom_nodes)-1
layers = 2*[[1, 5, 5, 1]]
num_elem = 10
num_gauss_pts = 8

k = 8
exact_sol = lambda x : jnp.sin(k*jnp.pi*x)
deriv_exact_sol = lambda x: k*jnp.pi*jnp.cos(k*jnp.pi*x)
rhs_fun = lambda x : k**2*jnp.pi**2*jnp.sin(k*jnp.pi*x)

coords = []
rhs = []
weights = []
# prepare the training data
for i in range(num_subdomains):
    x, w = generate_quad_pts_weights_1d(sub_dom_nodes[i], sub_dom_nodes[i+1],
                                        num_elem, num_gauss_pts)    
    x = jnp.expand_dims(x, 1)
    w = jnp.expand_dims(w, 1)
    f = rhs_fun(x)
    coords.append(x)
    weights.append(w)
    rhs.append(f)

x_bnd = jnp.array([xmin, xmax])
x_bnd = jnp.expand_dims(x_bnd, 1)
u_bnd = exact_sol(x_bnd)


model = Poisson1D_PU_DEM(key, layers, sub_dom_nodes, u_bnd)
#params = model.get_params(model.opt_state)
model.train(coords, rhs, weights, n_iter = 20000)


t_start = time.time()
params = model.get_params(model.opt_state)
loss_func = jax_tfp_function_factory(model,  params, coords, rhs, weights)
#with jax.disable_jit():
initial_pos = loss_func.init_params_1d
num_bfgs_iterations = 0
results = bfgs_minimize(loss_func, initial_position = initial_pos,
                    max_iterations=200)
new_pos = results.position
current_loss, _ = loss_func(new_pos)
num_bfgs_iterations += results.num_iterations
print("Iteration: ", num_bfgs_iterations, " loss: ", current_loss)
print("Time taken (BFGS) is ", time.time() - t_start)

_, unflatten_params = jax.flatten_util.ravel_pytree(params)
params = unflatten_params(results.position)

x_test_all = []
u_exact_all = []
du_exact_all = []
for i in range(num_subdomains):
    x_test = jnp.linspace(sub_dom_nodes[i], sub_dom_nodes[i+1], 2*len(coords[i]))
    x_test = jnp.expand_dims(x_test, 1)
    x_test_all.append(x_test)
    u_exact = exact_sol(x_test)
    u_exact_all.append(u_exact)
    du_exact = deriv_exact_sol(x_test)
    du_exact_all.append(du_exact)

u_pred_all = model.u_all(params, x_test_all)
du_pred_all = model.du_all(params, x_test_all)

for i in range(num_subdomains):
    plt.plot(x_test_all[i], u_pred_all[i], color='blue', label='Prediction')
    plt.plot(x_test_all[i], u_exact_all[i], color='orange', label='Ground truth')
    if i==0:
        plt.legend()
plt.title('u(x)')
plt.show()

for i in range(num_subdomains):
    plt.plot(x_test_all[i], du_pred_all[i], color='blue', label='Prediction')
    plt.plot(x_test_all[i], du_exact_all[i], color='orange', label='Ground truth')
    if i==0:
        plt.legend()
plt.title("u'(x)")
plt.show()


# %%
