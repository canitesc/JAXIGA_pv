#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:44:49 2024

@author: cosmin
"""
import jax.numpy as jnp

def energy_form_poisson_2d(R, dR, phys_pt, local_area, param_funs, aux_fields):
    a0 = param_funs[0]
    f = param_funs[1]
    energ = local_area * (a0(phys_pt[0], phys_pt[1]) * jnp.sum(dR**2) - f(phys_pt[0], phys_pt[1]) * R)
    return energ
