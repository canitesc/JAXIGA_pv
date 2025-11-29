#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:41:21 2023

@author: cosmin
"""
import jax
import jax.numpy as jnp
import time
from functools import partial
from jax import lax

from jaxiga.utils.misc import timing
from jaxiga.utils.bernstein import (bezier_extraction,
                             bernstein_basis,
                             bernstein_basis_deriv,
                             bernstein_basis_2d,
                             bernstein_basis_2nd_deriv)


def eval_spline_basis(u_hat, p2e, C, deg):
    """Evaluate the B-Spline basis

    Args:
        u_hat (list of floats): evaluation points
        p2e (array of integers): point to knot-span connectivity
        C (list of (deg+1)x(deg+1) 2D arrays): Bezier extraction matrices
        deg (integer): polynomial degree

    Returns:
        N (2D array of size (deg+1)*len(u_hat)): spline basis evaluated at u_hat
    """
    N = jnp.zeros((0, deg+1))
    for i in range(len(u_hat)):
        new_row = jnp.dot(C[int(p2e[i])], bernstein_basis(u_hat[i], deg).T).T
        N = jnp.concatenate((N, new_row), axis=0)
    return N


def eval_spline_basis_deriv(u_hat, uknots, p2e, C, deg):
    """
    Evaluates the B-spline basis derivatives

    Parameters
    ----------
    u_hat : (1D array of floats)
        evaluation points
    uknots : (1D array of floats)
        unique knot vector
    p2e : (array of integers)
        point to knot-span connectivity
    C : (list of (deg+1)x(deg+1) 2D arrays)
        Bezier extraction matrices
    deg : (integer)
        polynomial degree

    Returns
    -------
    dN : (2D array of size (deg+1)*len(u_hat))
        spline basis derivatives evaluated at u_hat

    """
    dN = jnp.zeros((0, deg+1))
    for i in range(len(u_hat)):
        k = int(p2e[i])
        jacob = 2/(uknots[k+1]-uknots[k])
        new_row = jnp.dot(C[k], bernstein_basis_deriv(u_hat[i], deg).T).T*jacob
        dN = jnp.concatenate((dN, new_row), axis=0)
    return dN


@timing
def eval_spline_basis_2nd_deriv(u_hat, uknots, p2e, C, deg):
    """
    Evaluates the 2nd derivatives of the B-spline basis

    Parameters
    ----------
    u_hat : (list of floats)
        evaluation points
    uknots : (1D array of floats)
        unique knot vector
    p2e : (array of integers)
        point to knot-span connectivity
    C : (list of (deg+1)x(deg+1) 2D arrays)
        Bezier extraction matrices
    deg : (integer)
        polynomial degree

    Returns
    -------
    ddN : (2D array of size (deg+1)*len(u_hat))
        2nd derivatives of the spline basis evaluated at u_hat

    """
    num_pts = len(u_hat)
    ddN = jnp.zeros((num_pts, deg+1))
    for i in range(num_pts):
        k = int(p2e[i])
        jacob = 4/(uknots[k+1]-uknots[k])**2
        new_row = jnp.dot(C[k], bernstein_basis_2nd_deriv(
            u_hat[i], deg).T).T*jacob
        ddN = ddN.at[i].set(new_row[0])
    return ddN

# @partial(jax.jit, static_argnums=(6,7,8))


def form_tensor_products_2d(Nu, Nv, dNu, dNv, ddNu, ddNv, num_pts_u, num_pts_v, deg):
    # initialize tensors for the 2D tensor-product B-spline basis and derivatives
    num_pts_uv = num_pts_u * num_pts_v
    M = jnp.einsum("ik,jl->jilk", Nu, Nv).reshape((num_pts_uv, (deg+1)**2))
    dMu = jnp.einsum("ik,jl->jilk", dNu, Nv).reshape((num_pts_uv, (deg+1)**2))
    dMv = jnp.einsum("ik,jl->jilk", Nu, dNv).reshape((num_pts_uv, (deg+1)**2))
    ddMu = jnp.einsum("ik,jl->jilk", ddNu,
                      Nv).reshape((num_pts_uv, (deg+1)**2))
    ddMv = jnp.einsum("ik,jl->jilk", Nu,
                      ddNv).reshape((num_pts_uv, (deg+1)**2))
    ddMuv = jnp.einsum("ik,jl->jilk", dNu,
                       dNv).reshape((num_pts_uv, (deg+1)**2))

    return M, dMu, dMv, ddMu, ddMv, ddMuv


@jax.jit
def compute_NURBS(M, dMu, dMv, ddMu, ddMv, ddMuv, cpts, wgts):
    M *= wgts
    dMu *= wgts
    dMv *= wgts
    ddMu *= wgts
    ddMv *= wgts
    ddMuv *= wgts

    w_sum = jnp.sum(M)
    dw_u = jnp.sum(dMu)
    dw_v = jnp.sum(dMv)
    d2w_u = jnp.sum(ddMu)
    d2w_v = jnp.sum(ddMv)
    d2w_uv = jnp.sum(ddMuv)

    ddMu = ddMu/w_sum - (2*dMu*dw_u + M*d2w_u)/w_sum**2 \
        + 2*M*dw_u**2/w_sum**3  # du^2 derivative
    ddMv = ddMv/w_sum - (2*dMv*dw_v + M*d2w_v)/w_sum**2 \
        + 2*M*dw_v**2/w_sum**3  # dv^2 derivative
    ddMuv = ddMuv/w_sum - (dMu*dw_v + dMv*dw_u + M*d2w_uv)/w_sum**2 \
        + 2*M*dw_u*dw_v/w_sum**3  # dudv derivative
    dMu = dMu/w_sum - M*dw_u/w_sum**2
    dMv = dMv/w_sum - M*dw_v/w_sum**2
    M = M/w_sum

    dM = jnp.stack((dMu, dMv))
    ddM = jnp.stack((ddMu, ddMuv, ddMv))
    coord = M[jnp.newaxis]@cpts
    dxdxi = dM@cpts
    d2xdxi2 = ddM@cpts

    dxdxi2 = jnp.array([[dxdxi[0, 0]**2, 2*dxdxi[0, 0]*dxdxi[0, 1], dxdxi[0, 1]**2],
                        [dxdxi[0, 0]*dxdxi[1, 0], dxdxi[0, 0]*dxdxi[1, 1] +
                            dxdxi[0, 1]*dxdxi[1, 0], dxdxi[0, 1]*dxdxi[1, 1]],
                        [dxdxi[1, 0]**2, 2*dxdxi[1, 0]*dxdxi[1, 1], dxdxi[1, 1]**2]])

    # solve for the first and second derivatives in global coordinates
    dM = jnp.linalg.solve(dxdxi, dM)
    ddM = jnp.linalg.solve(dxdxi2, ddM - d2xdxi2@dM)

    dMu = dM[0]
    dMv = dM[1]

    ddMu = ddM[0]
    ddMv = ddM[2]
    ddMuv = ddM[1]

    return M, dMu, dMv, ddMu, ddMv, ddMuv, coord

# @jax.jit
# @partial(jax.jit, static_argnums=(0,1))


def make_all_cpts_wgts(num_pts_uv, deg, p2e_uv, elem_node, cpts, wgts):
    cpts_all = jnp.zeros((num_pts_uv, (deg+1)**2, 2))
    wgts_all = jnp.zeros((num_pts_uv, (deg+1)**2))
    for i in range(num_pts_uv):
        elem_indx = p2e_uv[i][0]
        local_nodes = elem_node[elem_indx]
        wgts_all = wgts_all.at[i].set(wgts[local_nodes])
        cpts_all = cpts_all.at[i].set(cpts[0:2, local_nodes].transpose())

    return cpts_all, wgts_all


# @partial(jax.jit, static_argnums=(0, 1, 2))
def make_all_cpts_wgts_fem_2d(num_patches, num_elems, deg, elem_nodes, global_nodes,
                              cpts, wgts, Cs, num_fields):
    num_elem = sum(num_elems)
    num_nodes = (deg+1)**2
    cpts_all = jnp.zeros((num_elem, num_nodes, 2))
    wgts_all = jnp.zeros((num_elem, num_nodes))
    global_nodes_all = jnp.zeros((num_elem, num_fields*(deg+1)**2), dtype=int)
    C_all = jnp.stack(Cs)
    elem_indx = 0
    for i_patch in range(num_patches):
        local_nodes = elem_nodes[i_patch]
        cpts_all = cpts_all.at[elem_indx:elem_indx+num_elems[i_patch]].set(cpts[i_patch][0:2,
                                                                                         local_nodes].transpose(1, 2, 0))
        wgts_all = wgts_all.at[elem_indx:elem_indx +
                               num_elems[i_patch]].set(wgts[i_patch][local_nodes])
        global_nodes_all = global_nodes_all.at[elem_indx:elem_indx +
                                               num_elems[i_patch]].set(global_nodes[i_patch])
        elem_indx += num_elems[i_patch]

    return cpts_all, wgts_all, C_all, global_nodes_all


def pde_form_poisson_2d(RR, dR, phys_pt, local_area, param_funs):
    a0 = param_funs[0]
    f = param_funs[1]
    local_stiff = local_area * \
        a0(phys_pt[0], phys_pt[1]) * (dR.transpose() @ dR)
    local_rhs = local_area * f(phys_pt[0], phys_pt[1]) * RR
    return local_stiff, local_rhs


def pde_form_noop_2d(RR, dR, phys_pt, local_area, param_funs):
    num_nodes = len(RR)
    local_stiff = jnp.zeros((num_nodes, num_nodes))
    local_rhs = jnp.zeros_like(RR)
    return local_stiff, local_rhs


def pde_form_elast_2d(RR, dR, phys_pt, local_area, params):
    num_nodes = len(RR)
    Cmat = params[0].Cmat
    B = jnp.zeros((2 * num_nodes, 3))
    B = B.at[0: 2 * num_nodes - 1: 2, 0].set(dR[0, :])
    B = B.at[1: 2 * num_nodes: 2, 1].set(dR[1, :])
    B = B.at[0: 2 * num_nodes - 1: 2, 2].set(dR[1, :])
    B = B.at[1: 2 * num_nodes: 2, 2].set(dR[0, :])

    local_stiff = local_area * (B @ Cmat @ B.transpose())
    local_rhs = jnp.zeros(2*num_nodes)
    return local_stiff, local_rhs


def pde_form_elast_pf_2d(RR, dR, phys_pt, local_area, params):
    num_nodes = len(RR)
    Cmat = params[0].Cmat
    phi = params[0].phi
    B = jnp.zeros((2 * num_nodes, 3))
    B = B.at[0: 2 * num_nodes - 1: 2, 0].set(dR[0, :])
    B = B.at[1: 2 * num_nodes: 2, 1].set(dR[1, :])
    B = B.at[0: 2 * num_nodes - 1: 2, 2].set(dR[1, :])
    B = B.at[1: 2 * num_nodes: 2, 2].set(dR[0, :])

    local_stiff = local_area * (1-phi)**2*(B @ Cmat @ B.transpose())
    local_rhs = jnp.zeros(2*num_nodes)
    return local_stiff, local_rhs


def field_form_scalar_2d(RR, dR, local_sol_vals, params):
    local_field = jnp.dot(RR, local_sol_vals)

    return local_field


def _compute_basis_2d(ij_gauss, carry, C, Buv, dBdu, dBdv, u_diff, v_diff, cpts, wgts):
    i_gauss = ij_gauss[0]
    j_gauss = ij_gauss[1]

    # compute the B-spline basis functions and derivatives with
    # Bezier extraction
    N_mat = C @ Buv[i_gauss, j_gauss, :]
    dN_du = C @ dBdu[i_gauss, j_gauss, :] * 2 / u_diff
    dN_dv = C @ dBdv[i_gauss, j_gauss, :] * 2 / v_diff

    # compute the rational basis
    RR = N_mat * wgts
    dRdu = dN_du * wgts
    dRdv = dN_dv * wgts
    w_sum = jnp.sum(RR)
    dw_xi = jnp.sum(dRdu)
    dw_eta = jnp.sum(dRdv)

    dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
    dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2

    # compute the solution w.r.t. the physical space
    dR = jnp.stack((dRdu, dRdv))
    dxdxi = dR @ cpts
    dR = jnp.linalg.solve(dxdxi, dR)
    jac_par_phys = jnp.linalg.det(dxdxi)

    RR /= w_sum

    return RR, dR, jac_par_phys


def _compute_local_stiffs_2d(jac_ref_par, cpts, wgts, u_diff, v_diff, C,
                             Buv, dBdu, dBdv, wgts_u, wgts_v,  pde_form,
                             param_funs, carry, ij_gauss):
    i_gauss = ij_gauss[0]
    j_gauss = ij_gauss[1]

    local_stiff = carry[0]
    local_rhs = carry[1]
    elem_area = carry[2]

    RR, dR, jac_par_phys = _compute_basis_2d(ij_gauss, carry, C, Buv,
                                             dBdu, dBdv, u_diff, v_diff,
                                             cpts, wgts)
    phys_pt = RR @ cpts
    # print("phys_pt = ", phys_pt)
    # plt.scatter(phys_pt[0], phys_pt[1], s=0.5, c="black")
    local_area = (
        jac_par_phys * jac_ref_par * wgts_u[i_gauss] * wgts_v[j_gauss]
    )
    local_stiff_temp, local_rhs_temp = pde_form(
        RR, dR, phys_pt, local_area, param_funs)
    local_stiff += local_stiff_temp
    local_rhs += local_rhs_temp
    elem_area += local_area
    return ([local_stiff, local_rhs, elem_area], phys_pt)


def _compute_local_field_2d(jac_ref_par, cpts, wgts, u_diff, v_diff, C, sol_vals,
                            Buv, dBdu, dBdv, field_form, param_funs, carry, ij_gauss):

    RR, dR, _ = _compute_basis_2d(ij_gauss, carry, C, Buv,
                                             dBdu, dBdv, u_diff, v_diff, cpts, wgts)
    local_field = field_form(RR, dR, sol_vals, param_funs)

    return (None, local_field)


@partial(jax.jit, static_argnums=(6, 7, 13, 14, 15, 16))
def local_stiff_fun_2d(jac_ref_par, cpts, wgts, u_diff, v_diff, C,
                       num_gauss_u, num_gauss_v, Buv, dBdu, dBdv,
                       wgts_u, wgts_v, num_fields, pde_form, param_funs, aux_fields):
    # compute the rational spline basis
    num_nodes = len(wgts)
    local_stiff = jnp.zeros((num_fields*num_nodes, num_fields*num_nodes))
    local_rhs = jnp.zeros(num_fields*num_nodes)
    elem_area = 0
    ij_gausses = []
    for j_gauss in range(num_gauss_v):
        for i_gauss in range(num_gauss_u):
            ij_gausses.append([i_gauss, j_gauss])
    carry = [local_stiff, local_rhs, elem_area]

    _compute_local_partial = partial(_compute_local_stiffs_2d, jac_ref_par,
                                     cpts, wgts, u_diff, v_diff, C,
                                     Buv, dBdu, dBdv, wgts_u, wgts_v, pde_form,
                                     param_funs, aux_fields)

    carry, phys_pt = lax.scan(_compute_local_partial,
                              carry, jnp.array(ij_gausses))
    local_stiff = carry[0]
    local_rhs = carry[1]
    elem_area = carry[2]
    return local_stiff, local_rhs, elem_area, phys_pt


@partial(jax.jit, static_argnums=(7, 8, 12, 13, 14))
def local_field_2d(jac_ref_par, cpts, wgts, u_diff, v_diff, C, sol_vals,
                   num_gauss_u, num_gauss_v, Buv, dBdu, dBdv,
                   num_fields, field_form, params):
    # compute the rational spline basis
    ij_gausses = []
    for j_gauss in range(num_gauss_v):
        for i_gauss in range(num_gauss_u):
            ij_gausses.append([i_gauss, j_gauss])
    carry = None

    _compute_local_partial = partial(_compute_local_field_2d, jac_ref_par,
                                     cpts, wgts, u_diff, v_diff, C, sol_vals,
                                     Buv, dBdu, dBdv, field_form, params)

    carry, field_pt = lax.scan(
        _compute_local_partial, carry, jnp.array(ij_gausses))
    # jax.debug.print("field_pt = {}", field_pt)
    return field_pt


@timing
def evaluate_spline_basis_2d(patch_list, mesh_list):
    for patch, mesh in zip(patch_list, mesh_list):
        t = time.time()
        knot_u = patch.knot_u
        knot_v = patch.knot_v
        deg = mesh.deg[0]
        C_u, nb_u = bezier_extraction(knot_u, deg)
        C_v, nb_v = bezier_extraction(knot_v, deg)
        Nu = eval_spline_basis(patch.u_hat, patch.p2e_u, C_u, deg)
        dNu = eval_spline_basis_deriv(patch.u_hat, patch.uknots, patch.p2e_u,
                                      C_u, deg)
        ddNu = eval_spline_basis_2nd_deriv(
            patch.u_hat, patch.uknots, patch.p2e_u, C_u, deg)

        Nv = eval_spline_basis(patch.v_hat, patch.p2e_v, C_v, deg)
        dNv = eval_spline_basis_deriv(patch.v_hat, patch.vknots, patch.p2e_v,
                                      C_v, deg)
        ddNv = eval_spline_basis_2nd_deriv(patch.v_hat, patch.vknots, patch.p2e_v,
                                           C_v, deg)
        elapsed = time.time() - t
        print("Evaluating 1D B-Splines took ", elapsed, " seconds")

        t = time.time()

        M, dMu, dMv, ddMu, ddMv, ddMuv = form_tensor_products_2d(Nu, Nv, dNu,
                                                                 dNv, ddNu, ddNv,
                                                                 patch.num_pts_u,
                                                                 patch.num_pts_v, deg)
        elapsed = time.time() - t

        print("Evaluating tensor products took ", elapsed, " seconds")

        t = time.time()
        coords = jnp.zeros((patch.num_pts_uv, 2))
        cpts_all, wgts_all = make_all_cpts_wgts(patch.num_pts_uv, deg, patch.p2e_uv,
                                                mesh.elem_node, mesh.cpts,
                                                mesh.wgts)

        M, dMu, dMv, ddMu, ddMv, ddMuv, coords = jax.vmap(compute_NURBS)(M, dMu, dMv,
                                                                         ddMu, ddMv,
                                                                         ddMuv, cpts_all,
                                                                         wgts_all)

        elapsed = time.time() - t
        print("Evaluating NURBS took", elapsed, "seconds")
        mesh.coords = coords
        mesh.M = M
        mesh.dMu = dMu
        mesh.dMv = dMv
        mesh.ddMu = ddMu
        mesh.ddMv = ddMv
        mesh.ddMuv = ddMuv


@jax.jit
def make_triplet_array(global_nodes, local_stiff):
    num_nodes = len(global_nodes)
    II = jnp.tile(global_nodes, num_nodes)
    JJ = jnp.repeat(global_nodes, num_nodes)
    S = jnp.reshape(local_stiff, num_nodes ** 2)
    return II, JJ, S


@timing
def make_rhs(global_nodes_all, local_rhss, num_fields, size_basis):
    rhs = jnp.zeros(num_fields*size_basis)
    rhs = rhs.at[global_nodes_all].add(local_rhss)
    return rhs


def make_global_nodes_xy(global_nodes, num_fields):
    global_nodes_xy = []
    for global_node in global_nodes:
        global_node_xy = num_fields*global_node
        for i in range(num_fields-1):
            global_node_xy = jnp.stack(
                (global_node_xy, num_fields*global_node+i+1), axis=1)
        global_nodes_xy.append(jnp.reshape(
            global_node_xy, num_fields*len(global_node)))

    return global_nodes_xy


def extract_mesh_variables(mesh_list, num_fields):
    cpts = []
    wgts = []
    elem_nodes = []
    global_nodes = []
    num_elems = []
    Cs = []
    num_patches = len(mesh_list)
    jac_ref_pars = jnp.zeros((0))
    u_diffs = jnp.zeros((0))
    v_diffs = jnp.zeros((0))
    for i_patch in range(num_patches):
        cpts.append(mesh_list[i_patch].cpts)
        wgts.append(mesh_list[i_patch].wgts)
        elem_nodes.append(mesh_list[i_patch].elem_node)
        global_nodes_xy = make_global_nodes_xy(mesh_list[i_patch].elem_node_global,
                                               num_fields)
        global_nodes.append(global_nodes_xy)
        num_elems.append(mesh_list[i_patch].num_elem)
        u_mins = mesh_list[i_patch].elem_vertex[:, 0]
        u_maxs = mesh_list[i_patch].elem_vertex[:, 2]
        v_mins = mesh_list[i_patch].elem_vertex[:, 1]
        v_maxs = mesh_list[i_patch].elem_vertex[:, 3]
        u_diff = u_maxs - u_mins
        v_diff = v_maxs - v_mins

        jac_ref_par = u_diff * v_diff / 4
        jac_ref_pars = jnp.concatenate((jac_ref_pars, jac_ref_par))
        u_diffs = jnp.concatenate((u_diffs, u_diff))
        v_diffs = jnp.concatenate((v_diffs, v_diff))
        for i_elem in range(mesh_list[i_patch].num_elem):
            Cs.append(mesh_list[i_patch].C[i_elem])
    return cpts, wgts, elem_nodes, global_nodes, num_elems, Cs, u_diffs, v_diffs, jac_ref_pars


@timing
def evaluate_spline_basis_fem_2d(mesh_list, gauss_rule, num_fields, pde_form, param_funs, aux_fields):
    t = time.time()
    deg = mesh_list[0].deg
    pts_u = gauss_rule[0]["nodes"]
    pts_v = gauss_rule[1]["nodes"]
    wgts_u = gauss_rule[0]["weights"]
    wgts_v = gauss_rule[1]["weights"]
    num_gauss_u = len(pts_u)
    num_gauss_v = len(pts_v)
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, deg)
    elapsed = time.time() - t
    print("Evaluating 2D Bezier polynomials took", elapsed, "seconds")

    t = time.time()
    # Allocate memory for the triplet arrays
    (cpts, wgts, elem_nodes, global_nodes,
     num_elems, Cs, u_diffs, v_diffs, jac_ref_pars) = extract_mesh_variables(mesh_list,
                                                                             num_fields)

    elapsed = time.time() - t
    print("Extracting the mesh variables took ", elapsed, " seconds")
    num_patches = len(mesh_list)
    t = time.time()
    cpts_all, wgts_all, C_all, global_nodes_all = make_all_cpts_wgts_fem_2d(num_patches,
                                                                            tuple(
                                                                                num_elems), deg[0],
                                                                            elem_nodes, global_nodes, cpts,
                                                                            wgts, Cs, num_fields)
    elapsed = time.time() - t
    print("Making cpts and wgts arrays took", elapsed, " seconds")

    t = time.time()
    local_stiffs, local_rhss, elem_areas, phys_pts = jax.vmap(local_stiff_fun_2d,
                                                              in_axes=(0, 0, 0, 0, 0, 0,
                                                                       None, None, None,
                                                                       None, None, None,
                                                                       None, None, None,
                                                                       None, None))(jac_ref_pars,
                                                                              cpts_all,
                                                                              wgts_all,
                                                                              u_diffs,
                                                                              v_diffs,
                                                                              C_all,
                                                                              num_gauss_u,
                                                                              num_gauss_v,
                                                                              Buv,
                                                                              dBdu,
                                                                              dBdv,
                                                                              wgts_u,
                                                                              wgts_v,
                                                                              num_fields,
                                                                              pde_form,
                                                                              param_funs,
                                                                              aux_fields)
    print("Area of the domain is", jnp.sum(elem_areas))
    elapsed = time.time() - t
    print("Evaluating local stiffness matrices took", elapsed, " seconds")

    # make the triplet arrays
    t = time.time()

    IIs, JJs, Ss = jax.vmap(make_triplet_array)(global_nodes_all, local_stiffs)

    II = jnp.reshape(IIs, -1)
    JJ = jnp.reshape(JJs, -1)
    S = jnp.reshape(Ss, -1)

    elapsed = time.time() - t
    print("Making triplet arrays took", elapsed, " seconds")

    t = time.time()
    return II, JJ, S, local_rhss, global_nodes_all, phys_pts


def evaluate_field_fem_2d(mesh_list, gauss_rule, num_fields, field_form, sol_val, params):
    t = time.time()
    deg = mesh_list[0].deg
    pts_u = gauss_rule[0]["nodes"]
    pts_v = gauss_rule[1]["nodes"]
    num_gauss_u = len(pts_u)
    num_gauss_v = len(pts_v)
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, deg)
    elapsed = time.time() - t
    print("Evaluating 2D Bezier polynomials took", elapsed, "seconds")

    t = time.time()
    # Allocate memory for the triplet arrays
    (cpts, wgts, elem_nodes, global_nodes,
     num_elems, Cs, u_diffs, v_diffs, jac_ref_pars) = extract_mesh_variables(mesh_list,
                                                                             num_fields)

    elapsed = time.time() - t
    print("Extracting the mesh variables took ", elapsed, " seconds")
    num_patches = len(mesh_list)
    t = time.time()
    cpts_all, wgts_all, C_all, global_nodes_all = make_all_cpts_wgts_fem_2d(num_patches,
                                                                            tuple(
                                                                                num_elems), deg[0],
                                                                            elem_nodes, global_nodes, cpts,
                                                                            wgts, Cs, num_fields)
    elapsed = time.time() - t
    print("Making cpts and wgts arrays took", elapsed, " seconds")

    sol_vals_all = sol_val[global_nodes_all]

    t = time.time()
    field_vals = jax.vmap(local_field_2d,
                          in_axes=(0, 0, 0, 0, 0, 0, 0,
                                   None, None, None,
                                   None, None, None,
                                   None, None))(jac_ref_pars,
                                                cpts_all,
                                                wgts_all,
                                                u_diffs,
                                                v_diffs,
                                                C_all,
                                                sol_vals_all,
                                                num_gauss_u,
                                                num_gauss_v,
                                                Buv,
                                                dBdu,
                                                dBdv,
                                                num_fields,
                                                field_form,
                                                params)
    elapsed = time.time() - t
    print("Evaluating field values took", elapsed, " seconds")

    return field_vals
