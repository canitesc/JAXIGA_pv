#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimize the energy functionals corresponding to the 1D phase field method
with Deep Energy Method
@author: cosmin
"""

import jax
import jax.flatten_util
import jax.numpy as jnp
import itertools
import numpy as np
from functools import partial
from jax.example_libraries import optimizers
from jax.nn import swish
from jax import random, grad, vmap, jit, value_and_grad
from tqdm import trange
import matplotlib.pyplot as plt
from jax import config


key = jax.random.PRNGKey(42)
config.update("jax_enable_x64", True)

def generate_quad_pts_weights_1d(x_min=0, x_max=1, num_elem=10, num_gauss_pts=4):
    """
    Generates the Gauss points and weights on a 1D interval (x_min, x_max), split
    into num_elem equal-length subintervals, each with num_gauss_pts quadature
    points per element.
    
    Note: the sum of the weights should equal to the length of the domain
    Parameters
    ----------
    x_min : (scalar)
        lower bound of the 1D domain.
    x_max : (scalar)
        upper bound of the 1D domain.
    num_elem : (integer)
        number of subdivision intervals or elements.
    num_gauss_pts : (integer)
        number of Gauss points in each element
    Returns
    -------
    pts : (1D array)
        coordinates of the integration points.
    weights : (1D array)
        weights corresponding to each point.
    """
    x_pts = np.linspace(x_min, x_max, num=num_elem+1)
    pts = np.zeros(num_elem*num_gauss_pts)
    weights = np.zeros(num_elem*num_gauss_pts)
    pts_ref, weights_ref = np.polynomial.legendre.leggauss(num_gauss_pts)
    for i in range(num_elem):
        x_min_int = x_pts[i]
        x_max_int = x_pts[i+1]        
        jacob_int = (x_max_int-x_min_int)/2
        pts_int = jacob_int*pts_ref + (x_max_int+x_min_int)/2
        weights_int = jacob_int * weights_ref
        pts[i*num_gauss_pts:(i+1)*num_gauss_pts] = pts_int
        weights[i*num_gauss_pts:(i+1)*num_gauss_pts] = weights_int        
        
    return pts, weights

class PF1D_DEM:
    # base class for the 1D phase field model
    def __init__(self, rng_key, l0, layers, activation=swish, lr=3e-4):
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(self.init_layer, keys, layers[:-1], layers[1:]))
        self.activation = activation 
        self.l0 = l0
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
       
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


class PF1D_2nd_DEM(PF1D_DEM):
    # derived class for the 2nd order phase field
    def all_losses(self, params, coords, rhs, weights, bnd_inputs, bnd_targets):        
        u_pred = self.u(params, coords)
        dudx_pred = self.du_vec(params, coords)
        int_energy = (self.l0**2*dudx_pred**2 + u_pred**2)*jnp.squeeze(weights)
        bnd_pred = self.apply(params, bnd_inputs)
        bnd_residual = bnd_pred - bnd_targets
        int_loss = jnp.sum(int_energy)
        bnd_loss = jnp.mean(jnp.square(bnd_residual))
        return int_loss, bnd_loss
    
class PF1D_4th_DEM(PF1D_DEM):
    # derived class for the 4th order phase field
    def all_losses(self, params, coords, rhs, weights, bnd_inputs, bnd_targets):        
        u_pred = self.u(params, coords)
        dudx_pred = self.du_vec(params, coords)
        d2udx2_pred = self.d2u_vec(params, coords)
        int_energy = (self.l0**4/16*d2udx2_pred**2+self.l0**2/2*dudx_pred**2 +\
                       u_pred**2)*jnp.squeeze(weights)
        bnd_pred = self.apply(params, bnd_inputs)
        bnd_residual = bnd_pred - bnd_targets
        int_loss = jnp.sum(int_energy)
        bnd_loss = jnp.mean(jnp.square(bnd_residual))
        return int_loss, bnd_loss
    
class PF1D_modified_4th_DEM(PF1D_DEM):
    # derived class for the modified 4th order phase field
    def all_losses(self, params, coords, rhs, weights, bnd_inputs, bnd_targets):        
        u_pred = self.u(params, coords)
        dudx_pred = self.du_vec(params, coords)
        d2udx2_pred = self.d2u_vec(params, coords)
        int_energy = (self.l0**4/2**d2udx2_pred**2 + self.l0**2*dudx_pred**2 + \
                      u_pred**2)*jnp.squeeze(weights)
        bnd_pred = self.apply(params, bnd_inputs)
        bnd_residual = bnd_pred - bnd_targets
        int_loss = jnp.sum(int_energy)
        bnd_loss = jnp.mean(jnp.square(bnd_residual))
        return int_loss, bnd_loss


# define the problem
l0 = 0.25
exact_sol_4th = lambda x : jnp.exp(-jnp.abs(2*x)/l0)*(1+jnp.abs(2*x)/l0)
exact_sol_2nd = lambda x : jnp.exp(-jnp.abs(x)/l0)
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
x_test = jnp.linspace(x_min, x_max, 2*len(Xint))
x_test = jnp.expand_dims(x_test, 1)

# 2nd order phase field
u_exact_2nd = exact_sol_2nd(x_test)
model = PF1D_2nd_DEM(key, l0, [1, 16, 16, 1],lr=6e-4)
params = model.get_params(model.opt_state)

x_bnd = jnp.array([0.])
x_bnd = jnp.expand_dims(x_bnd, 1)
u_bnd = exact_sol_2nd(x_bnd)

#loss = model.loss(params, x, f, x_bnd, u_bnd)
model.train(Xint, Yint, Wint, x_bnd, u_bnd, n_iter = 40000)
params = model.get_params(model.opt_state)


u_pred = model.u(params, x_test)
u_exact_2nd = exact_sol_2nd(x_test)
u_exact_4th = exact_sol_4th(x_test)
err = jnp.squeeze(u_exact_4th) - u_pred
plt.plot(x_test, u_pred, label = 'Prediction')
#plt.plot(x_test, u_exact_4th, label = '4th order PF')
plt.plot(x_test, u_exact_2nd, label = 'Exact')

plt.title('2nd order phase field')
plt.legend()
plt.show()
print("L2-error norm: {}".format(jnp.linalg.norm(err)/jnp.linalg.norm(u_exact_2nd)))

plt.plot(err)
plt.title("Prediction error")
plt.show()

plt.plot(model.loss_log)
plt.title('Loss convergence')
plt.show()

# 4th order phase field
u_exact_4th = exact_sol_4th(x_test)
model = PF1D_4th_DEM(key, l0, [1, 16, 16, 1], lr=1e-3)
params = model.get_params(model.opt_state)

x_bnd = jnp.array([0.])
x_bnd = jnp.expand_dims(x_bnd, 1)
u_bnd = exact_sol_4th(x_bnd)

model.train(Xint, Yint, Wint, x_bnd, u_bnd, n_iter = 20000)
params = model.get_params(model.opt_state)

u_pred = model.u(params, x_test)
u_exact_4th = exact_sol_4th(x_test)
err = jnp.squeeze(u_exact_4th) - u_pred
plt.plot(x_test, u_pred, label = 'Prediction')
plt.plot(x_test, u_exact_4th, label = '4th order PF')

plt.title('4th order phase field')
plt.legend()
plt.show()
print("L2-error norm: {}".format(jnp.linalg.norm(err)/jnp.linalg.norm(u_exact_4th)))

plt.plot(err)
plt.title("Prediction error")
plt.show()

plt.plot(model.loss_log)
plt.title('Loss convergence')
plt.show()

# Modified 4th order phase field
model = PF1D_modified_4th_DEM(key, l0, [1, 16, 16, 1], lr=1e-4)
params = model.get_params(model.opt_state)

model.train(Xint, Yint, Wint, x_bnd, u_bnd, n_iter = 17500)
params = model.get_params(model.opt_state)

u_pred = model.u(params, x_test)
u_exact_4th = exact_sol_4th(x_test)
err = jnp.squeeze(u_exact_4th) - u_pred
plt.plot(x_test, u_pred, label = 'Prediction')
plt.plot(x_test, u_exact_4th, label = 'Exact')

plt.title('Modified 4th order phase field')
plt.legend()
plt.show()
print("L2-error norm: {}".format(jnp.linalg.norm(err)/jnp.linalg.norm(u_exact_4th)))

plt.plot(err)
plt.title("Prediction error")
plt.show()

plt.plot(model.loss_log)
plt.title('Loss convergence')
plt.show()
