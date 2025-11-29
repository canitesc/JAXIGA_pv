#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions related to the phase field model for fracture

@author: cosmin
"""
import jax.numpy as jnp
def history_edge_crack(phys_pts, B, l, cenerg):
    x = phys_pts[:, :, 0]
    y = phys_pts[:, :, 1]
    crack_tip_x = 0.5
    crack_tip_y = 0.5
    dis = jnp.where(x>0.5, jnp.sqrt((x-crack_tip_x)**2 + (y - crack_tip_y)**2),
                    jnp.abs(y-crack_tip_y))
    fenerg = jnp.where(dis<=l/2, B*cenerg*(1-dis/(l/2))/(2*l), 0)            
    return fenerg

def history_plate_w_3holes(phys_pts, B, l, cenerg):
    x = phys_pts[:, :, 0]
    y = phys_pts[:, :, 1]
    crack_tip_x = 4.
    crack_tip_y = 1.
    dis = jnp.where(y>1., jnp.sqrt((x-crack_tip_x)**2 + (y - crack_tip_y)**2),
                    jnp.abs(x-crack_tip_x))
    fenerg = jnp.where(dis<=l/2, B*cenerg*(1-dis/(l/2))/(2*l), 0)            
    return fenerg
    
def pos_strain_energy(material, strain_vals, fenerg):
    lam = material.lam
    mu = material.mu
    e1 = strain_vals[:,:,0]
    e2 = strain_vals[:,:,1]
    e3 = strain_vals[:,:,2]
    M = jnp.sqrt((e1-e2)**2 + e3**2) 
    lam_1 = 0.5*(e1+e2) + 0.5*M
    lam_2 = 0.5*(e1+e2) - 0.5*M
    pos_energ = 0.5*lam*(lam_1+lam_2)**2*jnp.where(lam_1+lam_2>0, 1., 0.) +\
        mu*(lam_1**2*jnp.where(lam_1>0, 1., 0.) + lam_2**2*jnp.where(lam_2>0, 1., 0.))
    pos_energ = jnp.maximum(pos_energ, fenerg)
    return pos_energ

def pos_strain_energy_DEM(material, strain_vals, fenerg, mask):
    lam = material.lam
    mu = material.mu
    e1 = strain_vals[:,:,0]
    e2 = strain_vals[:,:,1]
    e3 = strain_vals[:,:,2]
    temp = (e1-e2)**2 + e3**2 + 1e-10
    M = jnp.sqrt(temp)
    #M = jnp.where(temp>0, jnp.sqrt(temp), 0.)
    lam_1 = 0.5*(e1+e2) + 0.5*M
    lam_2 = 0.5*(e1+e2) - 0.5*M
    term_1 = 0.5*lam*(lam_1+lam_2)**2*jnp.where(lam_1+lam_2>0, 1., 0.)
    term_2 = mu*(lam_1**2*jnp.where(lam_1>0, 1., 0.))
    term_3 = mu*(lam_2**2*jnp.where(lam_2>0, 1., 0.))
                
    # term_1_a = 0.5*lam*(lam_1+lam_2)**2
    # term_2_a = mu*(lam_1**2)
    # term_3_a = mu*(lam_2**2)
                   
    
    #pos_energ = term_1_a + term_2_a + term_3_a     
    pos_energ = term_1 + term_2 + term_3    
    pos_energ = jnp.maximum(pos_energ, fenerg)
    pos_energ *= mask
    return pos_energ