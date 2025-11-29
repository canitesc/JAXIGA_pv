#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:32:31 2024

@author: cosmin
"""
import time
import itertools
import jax
import jax.numpy as jnp
from jax import lax
from jax import grad, jit, value_and_grad
from jax import config
import matplotlib.pyplot as plt
from jax.example_libraries import optimizers
from jax import random
from functools import partial
from tqdm import trange
from jaxiga.utils.jax_tfp_loss import jax_tfp_function_factory
from jaxiga.utils.bfgs import minimize as bfgs_minimize

key = random.PRNGKey(42)
# config.update("jax_enable_x64", True)

def diff_xx(y, num_pts, L):
    '''
    Compute the 2nd derivative with respect to x of function y whose values are 
    given on a grid with num_pts_x equally spaced points using the stencils:
    [1, -2, 1] in the interior
    [2, -5, 4, -1] at the left edge
    [-1, 4, -5, 2] at the right edge

    Parameters
    ----------
    y : (1d tensor)
        the function to be differentiated evaluated at uniformly spaced points on a grid.
    num_pts : (int)
        number of points in the x-direction 
    L : (float)
        length of the domain (in the x direction)    
    
    Returns
    -------
    y_xx : (1d tensor of length num_pts_x)
        the values of the xx-derivative evaluated at the points on the grid
    '''
    

    filter_xx = jnp.array([1., -2., 1.])[jnp.newaxis, jnp.newaxis, :]
    y_xx_left = 2*y[0]-5*y[1]+4*y[2]-1*y[3]
    y_xx_right = 2*y[-1]-5*y[-2]+4*y[-3]-1*y[-4]
    y = jnp.reshape(y, (1, 1, -1))
    y_xx = lax.conv(y, filter_xx, (1,), padding='SAME').squeeze()
    y_xx = y_xx.at[0].set(y_xx_left)
    y_xx = y_xx.at[-1].set(y_xx_right)
    y_xx *= (((num_pts-1)/L)**2)        
    return y_xx
    

class PF1D:
    '''
    Class for the Phase Field 1D problem
    '''
    def __init__(self, model_data, num_epoch, print_epoch, data_type):
        self.l0 = model_data['l0']                
        self.phi_bnd = model_data['phi_bnd']
        self.num_pts = model_data['n_points_left']+model_data['n_points_right']+ \
            len(self.phi_bnd)
        self.L = model_data['x_max'] - model_data['x_min']
        
        params_left = jnp.zeros(model_data['n_points_left'], dtype=data_type)
        params_right = jnp.zeros(model_data['n_points_right'], dtype=data_type)
        params = [params_left, params_right]
                
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-4)
        self.opt_state = self.opt_init(params)
        
        self.num_epoch = num_epoch
        self.print_epoch = print_epoch
        self.itercount = itertools.count()
        self.loss_log = []
        
    def get_phi(self, params):
        return jnp.concatenate((params[0], self.phi_bnd, params[1]))
        
    def get_loss(self, params):
        phi = self.get_phi(params)
        phi_xx = diff_xx(phi, self.num_pts, self.L)
        # remove the middle point from loss calcuations
        phi = jnp.concatenate((phi[:len(params[0])], phi[len(params[0])+1:]))
        phi_xx = jnp.concatenate((phi_xx[:len(params[0])], phi_xx[len(params[0])+1:]))
        loss = jnp.mean(jnp.square(self.l0**2*phi_xx - phi))
        return loss
        
    def get_loss_and_grads(self, params, *args):
        loss, grads = value_and_grad(self.get_loss)(params, *args)
        return loss, grads
           
      
    @partial(jit, static_argnums=(0,))    
    def step(self, i, opt_state, *args):
        params = self.get_params(opt_state)

        g = grad(self.get_loss)(params, *args)
        return self.opt_update(i, g, opt_state)
    
    def train(self, *args):
        pbar = trange(self.num_epoch)

        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       *args)
            if it % self.print_epoch == 0:
                params = self.get_params(self.opt_state)
    
                # Compute loss
                loss_value = self.get_loss(params, *args)
    
                # Store loss
                self.loss_log.append(loss_value)
    
                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
                
                

    
# Parameters
model_data = {}
model_data['l0'] = 0.25
model_data['x_min'] = -2.
model_data['x_max'] = 2.
model_data['x_bnd'] = jnp.array([0.])
model_data['phi_bnd'] = jnp.array([1.])
model_data['n_points_left'] = 100 
model_data['n_points_right'] = 100

num_epoch = 100000
print_epoch = 100
data_type = 'float32'

pred_model = PF1D(model_data, num_epoch, print_epoch, data_type)
params = pred_model.get_params(pred_model.opt_state)
t0 = time.time()
print("Training (ADAM)...")
pred_model.train()

t1 = time.time()
print("Time taken (ADAM)", t1-t0, "seconds")
t_start = time.time()
params = pred_model.get_params(pred_model.opt_state)
# loss_func = jax_tfp_function_factory(pred_model,  params)

# #with jax.disable_jit():
# initial_pos = loss_func.init_params_1d
# tolerance = 1e-5
# current_loss, _ = loss_func(initial_pos)
# num_bfgs_iterations = 0
# results = bfgs_minimize(loss_func, initial_position = initial_pos,
#                     max_iterations=10000)
# initial_pos = results.position
# num_bfgs_iterations += results.num_iterations
   
# print("Iteration: ", num_bfgs_iterations, " loss: ", current_loss)
# print("Time taken (BFGS) is ", time.time() - t_start)

# _, unflatten_params = jax.flatten_util.ravel_pytree(params)
# params = unflatten_params(results.position)
phi = pred_model.get_phi(params)
x_val = jnp.linspace(model_data['x_min'], model_data['x_max'], pred_model.num_pts)

exact_sol = lambda x : jnp.exp(-jnp.abs(x)/model_data['l0'])
phi_exact = exact_sol(x_val)
plt.plot(x_val, phi, label='Predicted')
plt.plot(x_val, phi_exact, label='Exact')
plt.legend()
plt.title('phi')
plt.show()

plt.plot(x_val, phi_exact-phi)
plt.title('Error')
plt.show()
