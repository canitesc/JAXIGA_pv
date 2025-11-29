#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solver classes for JAX

@author: cosmin
"""
import itertools
import jax
import jax.numpy as jnp
import random
from jax.example_libraries import optimizers
from functools import partial
from jax import grad, jit, value_and_grad, vmap
from tqdm import trange

from jaxiga.utils.processing_splines import (evaluate_field_fem_2d,
                                      field_form_scalar_deriv_2d,
                                      field_form_scalar_2d,
                                      field_form_strains_2d)
from jaxiga.utils.processing_splines_3d import (evaluate_field_fem_3d,
                                      field_form_scalar_deriv_3d,
                                      field_form_scalar_3d,
                                      field_form_strains_3d)
from jaxiga.utils.phase_field import pos_strain_energy_DEM



class Lin_Solver:
    def __init__(self, dim_space):

        self.itercount = itertools.count()
        params = jnp.zeros((dim_space,1))
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-4)
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []

    def loss(self, params, LHS, rhs):
        residual = LHS@params-rhs
        loss_val = jnp.mean(jnp.square(residual))#/jnp.linalg.norm(targets))
        return loss_val
    
    @partial(jit, static_argnums=(0,))
    def get_loss_and_grads(self, params, *args):
        loss_val, grads = value_and_grad(self.loss)(params, *args)        
        return loss_val, grads
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, LHS, rhs):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, LHS, rhs)
        return self.opt_update(i, g, opt_state)
    
    def train(self, LHS, rhs, n_iter = 1000):
        pbar = trange(n_iter)
        # Main training loop
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, LHS, rhs)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
    
                # Compute loss
                loss_value = self.loss(params, LHS, rhs)
                
                # # Store loss
                # self.loss_log.append(loss_value)
    
                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})

class Lin_Inverse:
    def __init__(self, dim_space):

        self.itercount = itertools.count()
        params = jnp.zeros((dim_space-2, dim_space-2))
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-7)
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []

    def loss(self, params, LHS, rhs):
        residual = LHS@params-rhs
        loss_val = jnp.sum(jnp.square(residual))#/jnp.linalg.norm(targets))
        return loss_val
    
    @partial(jit, static_argnums=(0,))
    def get_loss_and_grads(self, params, *args):
        loss_val, grads = value_and_grad(self.loss)(params, *args)        
        return loss_val, grads
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, LHS, rhs):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, LHS, rhs)
        return self.opt_update(i, g, opt_state)
    
    def train(self, LHS, rhs, n_iter = 100000):
        pbar = trange(n_iter)
        # Main training loop
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, LHS, rhs)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
    
                # Compute loss
                loss_value = self.loss(params, LHS, rhs)
                
                # # Store loss
                # self.loss_log.append(loss_value)
    
                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})


class Poisson1D_PINN_Spline:
    def __init__(self, IEN, p2e, deg, dim_space):
        self.deg = deg
        self.IEN = IEN
        self.p2e = p2e
        self.itercount = itertools.count()
        params = jnp.zeros((dim_space-2,1))
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-4)
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        # Logger
        self.loss_log = []

    def loss(self, params, ddNval, fvals):
        residual = jnp.zeros_like(fvals)
        coefs = jnp.concatenate((jnp.array([[0.]]), params, jnp.array([[0.]])))
        #use p2e
        for i in range(self.deg+1):
            residual += coefs[self.IEN[self.p2e[:,0],i]]*ddNval[:,i:i+1]
        residual += fvals            
        loss_val = jnp.mean(jnp.square(residual))
        return loss_val
    
    @partial(jit, static_argnums=(0,))
    def get_loss_and_grads(self, params, *args):
        loss_val, grads = value_and_grad(self.loss)(params, *args)        
        return loss_val, grads
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ddNval, fvals):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, ddNval, fvals)
        return self.opt_update(i, g, opt_state)
    
    def train(self, ddNval, fvals, n_iter = 1000):
        pbar = trange(n_iter)
        # Main training loop
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, ddNval, fvals)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
    
                # Compute loss
                loss_value = self.loss(params, ddNval, fvals)
                
                # # Store loss
                # self.loss_log.append(loss_value)
    
                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
                
class Poisson2D_PINN_Spline:
    def __init__(self, IEN, p2e, deg, size_basis, bcdof, bcval):
        self.deg = deg
        self.IEN = jnp.array(IEN)
        self.p2e = p2e
        self.itercount = itertools.count()
        num_free_dof = size_basis - len(bcdof)
        params = jnp.zeros((num_free_dof,1))
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-2)
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        self.bcdof = bcdof
        self.bcval = bcval[:, jnp.newaxis]
        
        # trainable coefs
        self.trainable_indx = jnp.setdiff1d(jnp.array(range(size_basis)), self.bcdof)
        self.index_map = jnp.zeros(size_basis, dtype=int)
        num_bcdofs = len(self.bcdof)
        self.index_map = self.index_map.at[self.bcdof].set(list(range(num_bcdofs)))
        self.index_map = self.index_map.at[self.trainable_indx].set(list(range(num_bcdofs,
                                                                               size_basis)))
        
        # Logger
        self.loss_log = []

    def loss(self, params, ddNx, ddNy, fvals):
        residual = jnp.zeros_like(fvals)
        coefs = jnp.concatenate((self.bcval, params))
        #use p2e
        num_basis = (self.deg+1)**2
        for i in range(num_basis):
            residual += coefs[self.index_map[self.IEN[self.p2e[:,0],i]]]*(ddNx[:,i:i+1] + \
                                                                          ddNy[:,i:i+1])
        residual += fvals            
        loss_val = jnp.mean(jnp.square(residual))
        return loss_val
    
    @partial(jit, static_argnums=(0,))
    def get_loss_and_grads(self, params, *args):
        loss_val, grads = value_and_grad(self.loss)(params, *args)        
        return loss_val, grads
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ddNx, ddNy, fvals):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, ddNx, ddNy, fvals)
        return self.opt_update(i, g, opt_state)
    
    def train(self, ddNx, ddNy, fvals, n_iter = 1000):
        pbar = trange(n_iter)
        # Main training loop
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, ddNx, ddNy, fvals)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
    
                # Compute loss
                loss_value = self.loss(params, ddNx, ddNy, fvals)
                
                # # Store loss
                # self.loss_log.append(loss_value)
    
                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
                
                
class Poisson2D_DEM_Spline:
    def __init__(self, global_nodes_all, deg, size_basis, bcdof, bcval):
        self.deg = deg
        self.itercount = itertools.count()
        num_free_dof = size_basis - len(bcdof)
        params = jnp.zeros((num_free_dof,1))
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-2)
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        self.bcdof = bcdof
        self.bcval = bcval[:, jnp.newaxis]
        
        # trainable coefs
        self.trainable_indx = jnp.setdiff1d(jnp.array(range(size_basis)), self.bcdof)
        self.index_map = jnp.zeros(size_basis, dtype=int)
        num_bcdofs = len(self.bcdof)
        self.index_map = self.index_map.at[self.bcdof].set(list(range(num_bcdofs)))
        self.index_map = self.index_map.at[self.trainable_indx].set(list(range(num_bcdofs,
                                                                               size_basis)))
        self.IEN = self.index_map[global_nodes_all]

        
        # Logger
        self.loss_log = []
           
    def loss(self, params, R, dR, fvals, local_areas):
        num_fields = 1
        coefs = jnp.concatenate((self.bcval, params))
        uh = evaluate_field_fem_2d(R, dR, num_fields, self.IEN,
                                            field_form_scalar_2d, coefs, ())
        duh = evaluate_field_fem_2d(R, dR, num_fields, self.IEN,
                                            field_form_scalar_deriv_2d, coefs, ())
        
        energ = jnp.sum((1/2*(duh[:,:,0,:].reshape(-1,1)**2 + duh[:,:,1,:].reshape(-1,1)**2) - \
                          fvals*uh.reshape(-1,1))*local_areas)
        
        return energ
    
    @partial(jit, static_argnums=(0,))
    def get_loss_and_grads(self, params, *args):
        loss_val, grads = value_and_grad(self.loss)(params, *args)        
        return loss_val, grads
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, *args):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, *args)
        return self.opt_update(i, g, opt_state)
    
    def train(self, *args, n_iter = 1000):
        pbar = trange(n_iter)
        
        #p2e_i = [self.IEN[self.p2e[:,0],i] for i in range(num_basis)]
        # Main training loop
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, *args)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
    
                # Compute loss
                loss_value = self.loss(params, *args)
                
                # # Store loss
                # self.loss_log.append(loss_value)
    
                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
                
class Poisson3D_DEM_Spline:
    def __init__(self, global_nodes_all, deg, size_basis, bcdof, bcval):
        self.deg = deg
        self.itercount = itertools.count()
        num_free_dof = size_basis - len(bcdof)
        params = jnp.zeros((num_free_dof,1))
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-2)
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        self.bcdof = bcdof
        self.bcval = bcval[:, jnp.newaxis]
        
        # trainable coefs
        self.trainable_indx = jnp.setdiff1d(jnp.array(range(size_basis)), self.bcdof)
        self.index_map = jnp.zeros(size_basis, dtype=int)
        num_bcdofs = len(self.bcdof)
        self.index_map = self.index_map.at[self.bcdof].set(list(range(num_bcdofs)))
        self.index_map = self.index_map.at[self.trainable_indx].set(list(range(num_bcdofs,
                                                                               size_basis)))
        self.IEN = self.index_map[global_nodes_all]

        
        # Logger
        self.loss_log = []
           
    def loss(self, params, R, dR, fvals, local_areas):
        num_fields = 1
        coefs = jnp.concatenate((self.bcval, params))
        uh = evaluate_field_fem_3d(R, dR, num_fields, self.IEN,
                                            field_form_scalar_3d, coefs, ())
        duh = evaluate_field_fem_3d(R, dR, num_fields, self.IEN,
                                            field_form_scalar_deriv_3d, coefs, ())
        
        energ = jnp.sum((1/2*(duh[:,:,0,:].reshape(-1,1)**2 + duh[:,:,1,:].reshape(-1,1)**2  + \
                          duh[:,:,2,:].reshape(-1,1)**2) - \
                          fvals*uh.reshape(-1,1))*local_areas)
        
        return energ
    
    @partial(jit, static_argnums=(0,))
    def get_loss_and_grads(self, params, *args):
        loss_val, grads = value_and_grad(self.loss)(params, *args)        
        return loss_val, grads
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, *args):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, *args)
        return self.opt_update(i, g, opt_state)
    
    def train(self, *args, n_iter = 1000):
        pbar = trange(n_iter)
        
        #p2e_i = [self.IEN[self.p2e[:,0],i] for i in range(num_basis)]
        # Main training loop
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, *args)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
    
                # Compute loss
                loss_value = self.loss(params, *args)
                
                # # Store loss
                # self.loss_log.append(loss_value)
    
                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
                
class Elast2D_DEM_Spline:
    def __init__(self, global_nodes_all, deg, size_basis, bcdof, bcval, material):
        self.deg = deg
        self.itercount = itertools.count()
        num_free_dof = 2*size_basis - len(bcdof)
        params = jnp.zeros((num_free_dof,1))
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-4)
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        self.bcdof = bcdof
        self.bcval = bcval[:, jnp.newaxis]
        self.Cmat = material.Cmat
        
        # trainable coefs
        self.trainable_indx = jnp.setdiff1d(jnp.array(range(2*size_basis)), self.bcdof)
        self.index_map = jnp.zeros(2*size_basis, dtype=int)
        num_bcdofs = len(self.bcdof)
        self.index_map = self.index_map.at[self.bcdof].set(list(range(num_bcdofs)))
        self.index_map = self.index_map.at[self.trainable_indx].set(list(range(num_bcdofs,
                                                                               2*size_basis)))
        self.IEN = self.index_map[global_nodes_all]

        # Logger
        self.loss_log = []                
   
    def loss(self, params, R, dR, local_areas_int, rhs):
        num_fields = 2
        coefs = jnp.concatenate((self.bcval, params))
        
        strain_vals = evaluate_field_fem_2d(R, dR, num_fields, self.IEN,
                                            field_form_strains_2d, jnp.squeeze(coefs), ()).reshape(-1,3)
        stress_vals = strain_vals@self.Cmat
        energ_int = jnp.sum(1/2*(strain_vals[:,0]*stress_vals[:,0] + \
                                strain_vals[:,1]*stress_vals[:,1] + \
                                strain_vals[:,2]*stress_vals[:,2])*local_areas_int)
            
        energ_ext = jnp.sum(jnp.squeeze(coefs)*rhs)
        return energ_int - energ_ext
    
    @partial(jit, static_argnums=(0,))
    def get_loss_and_grads(self, params, *args):
        loss_val, grads = value_and_grad(self.loss)(params, *args)        
        return loss_val, grads
    
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
                
                # # Store loss
                # self.loss_log.append(loss_value)
    
                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
                
class Elast3D_DEM_Spline:
    def __init__(self, global_nodes_all, deg, size_basis, bcdof, bcval, material):
        self.deg = deg
        self.itercount = itertools.count()
        num_free_dof = 3*size_basis - len(bcdof)
        params = jnp.zeros((num_free_dof,1))
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(1e-4)
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        self.bcdof = bcdof
        self.bcval = bcval[:, jnp.newaxis]
        self.Cmat = material.Cmat
        
        # trainable coefs
        self.trainable_indx = jnp.setdiff1d(jnp.array(range(3*size_basis)), self.bcdof)
        self.index_map = jnp.zeros(3*size_basis, dtype=int)
        num_bcdofs = len(self.bcdof)
        self.index_map = self.index_map.at[self.bcdof].set(list(range(num_bcdofs)))
        self.index_map = self.index_map.at[self.trainable_indx].set(list(range(num_bcdofs,
                                                                               3*size_basis)))
        self.IEN = self.index_map[global_nodes_all]

        # Logger
        self.loss_log = []                
   
    def loss(self, params, R, dR, local_areas_int, rhs):
        num_fields = 3
        coefs = jnp.concatenate((self.bcval, params))
        
        strain_vals = evaluate_field_fem_3d(R, dR, num_fields, self.IEN,
                                            field_form_strains_3d, jnp.squeeze(coefs), ()).reshape(-1,6)
        stress_vals = strain_vals@self.Cmat
        energ_int = jnp.sum(1/2*(strain_vals[:,0]*stress_vals[:,0] + \
                                strain_vals[:,1]*stress_vals[:,1] + \
                                strain_vals[:,2]*stress_vals[:,2] + \
                                strain_vals[:,3]*stress_vals[:,3] + \
                                strain_vals[:,4]*stress_vals[:,4] + \
                                strain_vals[:,5]*stress_vals[:,5] + \
                                strain_vals[:,6]*stress_vals[:,6])*local_areas_int)
            
        energ_ext = jnp.sum(jnp.squeeze(coefs)*rhs)
        return energ_int - energ_ext
    
    @partial(jit, static_argnums=(0,))
    def get_loss_and_grads(self, params, *args):
        loss_val, grads = value_and_grad(self.loss)(params, *args)        
        return loss_val, grads
    
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
                
                # # Store loss
                # self.loss_log.append(loss_value)
    
                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
                
class PF2D_DEM_Spline:
    def __init__(self, global_nodes_u, global_nodes_phi, deg, size_basis, bcdof,
                 material, l, cenerg, mask):
        self.deg = deg
        self.itercount = itertools.count()
        num_free_dof = 3*size_basis - len(bcdof)
        params = jnp.zeros((num_free_dof,1))
        # see https://twitter.com/karpathy/status/801621764144971776
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(3e-4)
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        self.bcdof = bcdof
        self.Cmat = material.Cmat
        self.material = material
        self.l = l
        self.cenerg = cenerg
        self.mask = mask
        
        # trainable coefs
        self.trainable_indx = jnp.setdiff1d(jnp.array(range(2*size_basis)), self.bcdof)
        self.index_map = jnp.zeros(2*size_basis, dtype=int)
        num_bcdofs = len(self.bcdof)
        self.index_map = self.index_map.at[self.bcdof].set(list(range(num_bcdofs)))
        self.index_map = self.index_map.at[self.trainable_indx].set(list(range(num_bcdofs,
                                                                               2*size_basis)))
        self.IEN_u = self.index_map[global_nodes_u]
        self.IEN_phi = global_nodes_phi

        # Logger
        self.loss_log = []
        
    def get_all_losses(self, params, R, dR, local_areas_int, bcval, fenerg):
        num_fields = 2
        num_free_u = len(self.index_map) - len(self.bcdof)
        coefs_u = jnp.concatenate((bcval, params[:num_free_u]))
        coefs_phi = params[num_free_u:]
        
        strain_vals = evaluate_field_fem_2d(R, dR, num_fields, self.IEN_u,
                            field_form_strains_2d, jnp.squeeze(coefs_u), ())
        
        fenerg = pos_strain_energy_DEM(self.material, strain_vals, fenerg, self.mask)
        
        strain_vals = strain_vals.reshape(-1,3)
        stress_vals = strain_vals@self.Cmat
        
        phi_vals = evaluate_field_fem_2d(R, dR, num_fields, self.IEN_phi,
                         field_form_scalar_2d, jnp.squeeze(coefs_phi), ()).flatten()
        phi_vals *= self.mask.flatten()

        phi_deriv_vals = evaluate_field_fem_2d(R, dR, num_fields, self.IEN_phi,
                         field_form_scalar_deriv_2d, jnp.squeeze(coefs_phi), ())
        g = (1-phi_vals)**2
        
        phi_x = phi_deriv_vals[:, :, 0].flatten()
        phi_y = phi_deriv_vals[:, :, 1].flatten()
        
        nabla = phi_x**2 + phi_y**2
        
        energ_dens = 1/2*g.flatten()*(strain_vals[:,0]*stress_vals[:,0] + \
                                strain_vals[:,1]*stress_vals[:,1] + \
                                strain_vals[:,2]*stress_vals[:,2])
        
        energ_int = jnp.sum(energ_dens*local_areas_int)
            
        phi_dens = 0.5*self.cenerg * (phi_vals**2/self.l + self.l*nabla) + \
                             g.flatten()*fenerg.flatten()

        # energy_phi = jnp.sum(phi_dens*local_areas_int)
        

        
        energy_phi_1 = jnp.sum((0.5*self.cenerg * (phi_vals**2/self.l + self.l*nabla))*local_areas_int)
        energy_phi_2 = jnp.sum(g.flatten()*fenerg.flatten()*local_areas_int)
                            
                              #g.flatten()*fenerg.flatten())*local_areas_int)
            #g.flatten()*fenerg.flatten())*local_areas_int)
            
        #jax.debug.print('energ_int = {x}, energ_ext = {y}', x=energ_int, y=energ_ext)
        return energ_int, energy_phi_1, energy_phi_2, fenerg, phi_dens, energ_dens
    
    def loss(self, *args):
        energ_int, energy_phi_1, energy_phi_2, _, _, _ = self.get_all_losses(*args)
        return energ_int + energy_phi_1 + energy_phi_2
    
    @partial(jit, static_argnums=(0,))
    def get_loss_and_grads(self, params, *args):
        loss_val, grads = value_and_grad(self.loss)(params, *args)        
        return loss_val, grads
    
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
                
                # # Store loss
                # self.loss_log.append(loss_value)
    
                # Print loss during training
                pbar.set_postfix({'Loss': loss_value})
