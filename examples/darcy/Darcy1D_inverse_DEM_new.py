#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Darcy 1D example with Deep Energy Method -> inverse problem

@author: cosmin
"""

import jax
import jax.flatten_util
import jax.numpy as jnp
import itertools
import time
from functools import partial
from jax.example_libraries import optimizers
from jax.nn import tanh, relu
from jax import random, grad, vmap, jit, value_and_grad
from tqdm import trange
import matplotlib.pyplot as plt
from jax import config

from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils.preprocessing_DEM_1d import generate_quad_pts_weights_1d
from jaxiga.utils.bfgs import minimize as bfgs_minimize


key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

class Darcy1D_DEM:
    def __init__(self, rng_key, layers, activation=tanh):
        key, *keys = random.split(rng_key, len(layers))        
        params_u = list(map(self.init_layer, keys, layers[:-1], layers[1:]))
        self.a = 8.
        self.activation = activation 
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-3)
       
        self.opt_state_u = self.opt_init(params_u)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []

    def init_layer(self, key, d_in, d_out):
        k1, k2 = random.split(key)
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b
    
    def apply(self, params_u, inputs):
        for W, b in params_u[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = self.activation(outputs)
        W, b = params_u[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs
    
    def u(self, params_u, x):
        u_val = jnp.squeeze(self.apply(params_u, x))
        return u_val
    
    def du(self, params_u, x):
        dudx = jnp.squeeze(grad(self.u, 1)(params_u, x))
        return dudx
    
    def du_vec(self, params_u, inputs):
        dudx = vmap(self.du, (None, 0))(params_u, inputs)
        return dudx                
    
    def all_losses(self, params_u, a, coords, rhs, weights, bnd_inputs, bnd_targets,
                   x_data, u_data):
        #preds = self.apply(params, inputs)
        u_pred = self.u(params_u, coords)
        dudx_pred = self.du_vec(params_u, coords)
        int_energy = (1/2*a*dudx_pred**2 - u_pred*jnp.squeeze(rhs))*jnp.squeeze(weights)
        bnd_pred = self.apply(params_u, bnd_inputs)
        bnd_residual = bnd_pred - bnd_targets
        data_pred = self.apply(params_u, x_data)
        data_residual = data_pred - u_data
        int_loss = jnp.sum(int_energy)
        bnd_loss = jnp.mean(jnp.square(bnd_residual))
        data_loss = jnp.mean(jnp.square(data_residual))
        return int_loss, bnd_loss, data_loss
    
    def loss_u(self, params_u, a, *args):
        int_loss, bnd_loss, data_loss = self.all_losses(params_u, a, *args)
        loss = int_loss + 1e3*bnd_loss #+ data_loss
        return loss
    
    def loss_a(self, params_u, a, *args):
        int_loss, bnd_loss, data_loss = self.all_losses(params_u, a, *args)
        return int_loss + bnd_loss #+ data_loss
    
    def get_loss_and_grads_u(self, params_u, a, *args):
        loss, grads = value_and_grad(self.loss_u)(params_u, a, *args)
        return loss, grads
    
    def get_loss_and_grads_a(self, params_u, a, *args):
        loss, grads = value_and_grad(self.loss_a, argnums=1)(params_u, a, *args)
        return loss, grads
    
    @partial(jit, static_argnums=(0,))    
    def step_u(self, i, opt_state_u, a, *args):
        params_u = self.get_params(opt_state_u)
        g = grad(self.loss_u)(params_u, a, *args)
        return self.opt_update(i, g, opt_state_u)
    
    def step_a(self, data_pred, u_data):
        norm_ratio = jnp.linalg.norm(data_pred)/jnp.linalg.norm(u_data)
        return norm_ratio
        
    
    def train_u(self, coords, rhs, weights, bnd_inputs, bnd_targets,
                   x_data, u_data, n_iter = 1000):
        pbar = trange(n_iter)
        # Main training loop
        for it in pbar:
            self.opt_state_u = self.step_u(next(self.itercount), self.opt_state_u,
                                           self.a,  coords, rhs, weights, bnd_inputs,
                                           bnd_targets, x_data, u_data)
            if it % 100 == 0:
                params_u = self.get_params(self.opt_state_u)
                data_pred = self.u(params_u, x_data)
                norm_ratio = self.step_a(data_pred, u_data)
                self.a += self.a*(norm_ratio-1)*0.1

                # Compute loss
                int_loss, bnd_loss, data_loss = self.all_losses(params_u, self.a, 
                                coords, rhs, weights, bnd_inputs, bnd_targets,
                                x_data, u_data)
                loss_value = int_loss + 1e3*bnd_loss #+ data_loss
                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss': loss_value, 'Bnd_loss': bnd_loss, 
                                  'Data_loss': data_loss, 'a': self.a})
                

# define the problem
a = 4.
exact_sol = lambda x : jnp.sin(2*jnp.pi*x)
rhs_fun = lambda x : a*4*jnp.pi**2*jnp.sin(2*jnp.pi*x)
           
#define the input and output data set
x_min = -1.
x_max = 1.
num_elem = 100
num_gauss_pts = 4
data_type = "float32"

Xint, Wint = generate_quad_pts_weights_1d(x_min, x_max, num_elem, num_gauss_pts)
Xint = jnp.array(Xint)[jnp.newaxis].T.astype(data_type)
Yint = rhs_fun(Xint)
Wint = jnp.array(Wint)[jnp.newaxis].T.astype(data_type)


num_data_pts = 10
x_data = jnp.linspace(x_min, x_max, num_data_pts+2)[1:-1]
x_data = jnp.expand_dims(x_data, 1)
u_data = exact_sol(x_data)


a_vals = jnp.arange(0.5, 10, 0.5)
losses = []
int_losses =[]
data_losses = []
model = Darcy1D_DEM(key, [1, 40, 40, 1])
params_u = model.get_params(model.opt_state_u)


rhs = rhs_fun(Xint)
x_bnd = jnp.array([x_min, x_max])
x_bnd = jnp.expand_dims(x_bnd, 1)
u_bnd = exact_sol(x_bnd)

#loss = model.loss(params, x, f, x_bnd, u_bnd)
model.train_u(Xint, Yint, Wint, x_bnd, u_bnd, x_data, u_data, n_iter = 20000)
#model.train_a(Xint, Yint, Wint, x_bnd, u_bnd, x_data, u_data, n_iter = 20000)

params_u = model.get_params(model.opt_state_u)
a = model.a

x_test = jnp.linspace(x_min, x_max, 10*len(Xint))
x_test = jnp.expand_dims(x_test, 1)
u_pred = model.u(params_u, x_test)
u_exact = exact_sol(x_test)
err = jnp.squeeze(u_exact) - u_pred
plt.plot(x_test, u_pred, label = 'Prediction')
plt.plot(x_test, u_exact, label = 'Ground truth')
plt.title('u(x)')
plt.legend()
plt.show()
print("L2-error norm: {}".format(jnp.linalg.norm(err)/jnp.linalg.norm(u_exact)))

