#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for spline-preprocessing

@author: cosmin
"""
import jax.numpy as jnp
from jaxiga.utils.misc import timing


@timing
def make_IEN(knots, deg):
    TOL = 1e-10
    IEN = jnp.zeros((0, deg+1), dtype=int)
    for i in range(len(knots)-1):
        if knots[i+1] - knots[i] > TOL:
            IEN = jnp.vstack((IEN, jnp.arange(i-deg, i+1)))
    return IEN

@timing
def make_Greville(knots, deg):
    G = jnp.zeros((0, 1))
    for i in range(len(knots)-deg-1):
        G = jnp.vstack((G, jnp.sum(knots[i+1:i+deg+1])/deg))
    return G



def find_index(S, x):
    # check if x is outside the range of S
    if x < S[0] or x > S[-1]:
        return None
    
    # use binary search to find the index N
    left, right = 0, len(S) - 1
    while left <= right:
        mid = (left + right) // 2
        if S[mid] <= x <= S[mid+1]:
            return mid
        elif x < S[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return None

@timing
def make_point_to_element_connectivity(G, uknots, TOL=1e-10):
    find_index_inner = lambda x : find_index(uknots, x)
    p2e = jnp.zeros_like(G, dtype=int)
    for i in range(len(G)):
        p2e = p2e.at[i].set(find_index_inner(G[i]))
    return p2e

@timing
def make_uknots(knots, TOL=1e-10):
    uknots = [knots[0]]
    for i in range(len(knots)-1):
        if knots[i+1]-knots[i]>TOL:
            uknots.append(knots[i+1])
    return jnp.array(uknots)

def map_parameter_to_reference(G, uknots, p2e):
    num_pts = len(G)    
    xhat = jnp.zeros((num_pts, 1))
    for i in range(num_pts):
        xhat = xhat.at[i].set((2*(G[i] - uknots[p2e[i]])/(uknots[p2e[i]+1] -\
                                                            uknots[p2e[i]]) - 1))
    return xhat

def get_bcdof(bound_cond, mesh_list):
    bcdof = []
    for i in range(len(bound_cond)):
        patch_index = bound_cond[i].patch_index
        side = bound_cond[i].side
        if bound_cond[i].type == "Dirichlet":
            bcdof += mesh_list[patch_index].bcdof_global[side]
    bcdof = jnp.unique(jnp.array(bcdof))
    return bcdof

def get_boundary_indices(num_u, num_v):
    down = jnp.array(range(0, num_u))
    left = jnp.array(range(0,(num_v-1)*num_u+1, num_u))
    up = jnp.array(range((num_v-1)*num_u, num_u*num_v))
    right = jnp.array(range(num_u-1, num_u*num_v, num_u))
    all_sides = jnp.unique(jnp.concatenate((down, left, right, up)))
    bound_indices = {'down' : down, 'left' : left, 'up' : up, 'right' : right, 
                     'all' : all_sides}
    return bound_indices

def get_interior_indices(bcdof, num_pts_uv, pt_bound_indices):
    
    interior_indices = jnp.setdiff1d(jnp.array(range(num_pts_uv)), bcdof)
    pt_interior_indices = jnp.setdiff1d(jnp.array(range(num_pts_uv)), pt_bound_indices['all'])
    return pt_interior_indices, interior_indices

def generate_Greville_abscissae2D(patch_list):
    for patch in patch_list:
        knot_u = jnp.array(patch.surf.knotvector_u)
        knot_v = jnp.array(patch.surf.knotvector_v)
        G_u = make_Greville(knot_u, patch.surf.degree_u)
        G_v = make_Greville(knot_v, patch.surf.degree_v)
    
        # Make uknots
        uknots = make_uknots(knot_u)
        vknots = make_uknots(knot_v)
        assert (uknots == jnp.unique(knot_u)).all()
        assert (vknots == jnp.unique(knot_v)).all()
    
        # Point to knot-span connectivity
        p2e_u = make_point_to_element_connectivity(G_u, uknots)
        p2e_v = make_point_to_element_connectivity(G_v, vknots)
    
        # Map the Greville abscissae to the reference element
        u_hat = map_parameter_to_reference(G_u, uknots, p2e_u)
        v_hat = map_parameter_to_reference(G_v, vknots, p2e_v)
    

        num_pts_u = len(G_u)
        num_pts_v = len(G_v)
        num_pts_uv = num_pts_u * num_pts_v
        G_uv = jnp.zeros((num_pts_uv, 2))
        index_counter = 0
        for j in range(num_pts_v):
            for i in range(num_pts_u):
                G_uv = G_uv.at[index_counter, 0].set(G_u[i,0])
                G_uv = G_uv.at[index_counter, 1].set(G_v[j,0])
                index_counter += 1

        patch.G_uv = G_uv
        patch.G_u = G_u
        patch.G_v = G_v
        patch.num_pts_u = num_pts_u
        patch.num_pts_v = num_pts_v
        patch.num_pts_uv = num_pts_uv
        patch.uknots = uknots
        patch.vknots = vknots
        patch.knot_u = knot_u
        patch.knot_v = knot_v
        patch.u_hat = u_hat
        patch.v_hat = v_hat
        patch.p2e_u = p2e_u
        patch.p2e_v = p2e_v
        
def make_point_to_element_connectivity_2d(patch_list):
    for patch in patch_list:
        uknots = patch.uknots
        vknots = patch.vknots
        # Point to knot-span connectivity
        p2e_u = make_point_to_element_connectivity(patch.G_u, uknots)
        p2e_v = make_point_to_element_connectivity(patch.G_v, vknots)
    
        p2e_uv = jnp.zeros((patch.num_pts_uv, 1), dtype=int)
    
        num_elem_u = len(uknots)-1
        index_counter = 0
        for j in range(patch.num_pts_v):
            for i in range(patch.num_pts_u):
                elem_indx_u = p2e_u[i,0]
                elem_indx_v = p2e_v[j,0]
                elem_indx = elem_indx_v*num_elem_u + elem_indx_u
                p2e_uv = p2e_uv.at[index_counter, 0].set(elem_indx)
                index_counter += 1
        patch.p2e_uv = p2e_uv

def get_mesh_elem_range(meshes):
    # returns the range of element indices for each patch/mesh
    start_elem = 0
    elem_range = []
    for mesh in meshes:
        mesh_range = [start_elem, start_elem+mesh.num_elem]
        elem_range.append(mesh_range)
        start_elem += mesh.num_elem
    return elem_range
