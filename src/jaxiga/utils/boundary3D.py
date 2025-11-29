#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class and functions for boundary conditions
"""
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning

from jaxiga.utils.bernstein import bernstein_basis_3d


class boundary3D:
    """
    Class for a boundary condition on a single edge in parameter space

    Input
    ------
    bnd_type : (string) type of boundary conditions (e.g. "Dirichlet")
    patch_index : (int) index of the patch in a multipatch mesh
    side : (string) side of the parameters space (e.g. "down", "right", "up",
                                                          "left", "front", "back")
    op_value : (function) value of the boundary function
    alpha: (number) parameter value for Robin boundary condition (\alpha u  + u')
    """

    def __init__(self, bnd_type, patch_index, side, op_value, alpha=0.0):
        self.type = bnd_type
        self.patch_index = patch_index
        self.side = side
        self.op_value = op_value
        self.alpha = alpha


def applyBC3D(mesh_list, bound_cond, lhs, rhs, quad_rule=None):
    """
    Applies the boundary conditions to a linear system for scalar problems
    TODO: At the moment, only homogeneous Dirichlet B.C. are implemented

    Parameters
    ----------
    mesh_list :(list of IGAMesh3D) multipatch mesh
    bound_cond : (list of boundary3D) boundary conditions
    lhs : (2D array) stiffness matrix
    rhs : (1D array) rhs vector
    quad_rule : (optional, list of dicts) list of Gauss points and weights in
                the reference interval [-1,1] (one for each parametric direction)

    Returns
    -------
    lhs: updated stiffness matrix
    rhs: updated rhs vector
    """
    bcdof = []

    for i in range(len(bound_cond)):
        patch_index = bound_cond[i].patch_index
        side = bound_cond[i].side
        if bound_cond[i].type == "Dirichlet":
            bcdof += mesh_list[patch_index].bcdof_global[side]
        elif bound_cond[i].type == "Neumann":
            op_val = bound_cond[i].op_value
            if side == "down" or side == "up":
                quad_rule_side = quad_rule[0]
            elif side == "left" or side == "right":
                quad_rule_side = quad_rule[1]
            rhs = applyNeumannScalar3D(
                mesh_list[patch_index], rhs, side, quad_rule_side, op_val
            )
        elif bound_cond[i].type == "Robin":
            op_val = bound_cond[i].op_value
            alpha = bound_cond[i].alpha
            if side == "down" or side == "up":
                quad_rule_side = quad_rule[0]
            elif side == "left" or side == "right":
                quad_rule_side = quad_rule[1]
            lhs, rhs = applyRobinScalar3D(
                mesh_list[patch_index], lhs, rhs, side, quad_rule_side, alpha, op_val
            )

    if len(bcdof) > 0:
        bcdof = np.unique(bcdof)
        bcval = np.zeros_like(bcdof)
        rhs = rhs - lhs[:, bcdof] * bcval
        rhs[bcdof] = bcval
        # TODO: fix this warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            lhs[bcdof, :] = 0
            lhs[:, bcdof] = 0
            lhs[bcdof, bcdof] = 1.0

    return lhs, rhs


def applyNeumannScalar3D(mesh, rhs, side, quad_rule_side, op_val):
    """
    !!!!!!!!!!!!! Not yet converted to 3D!!!!!!!!!!!!!!!
    Applies the Neumann boundary conditins on the given patch and side for
    scalar problems and updates the rhs vector

    Parameters
    ----------
    mesh :(IGAMesh2D) patch mesh
    rhs : (1D array) rhs vector
    side : (string) either "down" (v=0), "right" (u=1), "up" (v=1), "left" (u=0)
    quad_rule_side : (dict) auss points and weights in the reference
                interval [-1,1]
    op_val : (function) function for the flux

    Returns
    -------
    rhs : (1D array)
        updated rhs vector

    """
    index_all = mesh._get_boundary_indices()
    if side == "right":
        pts_u = np.array([1.0])
        pts_v = quad_rule_side["nodes"]
    elif side == "left":
        pts_u = np.array([-1.0])
        pts_v = quad_rule_side["nodes"]
    elif side == "down":
        pts_u = quad_rule_side["nodes"]
        pts_v = np.array([-1.0])
    elif side == "up":
        pts_u = quad_rule_side["nodes"]
        pts_v = np.array([1.0])
    sctr = index_all[side]
    # Form the 2D tensor product of the basis functions
    Buv, dBdu, dBdv = bernstein_basis_3d(pts_u, pts_v, mesh.deg)

    # Evaluate the Neumann integral on each element
    boundary_length = 0.0
    for i_elem in mesh.elem[side]:
        u_min = mesh.elem_vertex[i_elem, 0]
        u_max = mesh.elem_vertex[i_elem, 2]
        v_min = mesh.elem_vertex[i_elem, 1]
        v_max = mesh.elem_vertex[i_elem, 3]
        if side == "right" or side == "left":
            jac_par_ref = (v_max - v_min) / 2
        else:
            jac_par_ref = (u_max - u_min) / 2

        # compute the rational spline basis
        local_nodes = mesh.elem_node[i_elem]
        global_nodes = mesh.elem_node_global[i_elem]

        cpts = mesh.cpts[0:2, local_nodes]
        wgts = mesh.wgts[local_nodes]
        local_rhs = np.zeros(len(sctr), dtype=np.complex128)
        for i_gauss in range(len(quad_rule_side["nodes"])):
            # compute the (B-)spline basis functions and derivatives with Bezier extraction
            if side == "right" or side == "left":
                N_mat = mesh.C[i_elem] @ Buv[0, i_gauss, :]
                dN_du = mesh.C[i_elem] @ dBdu[0, i_gauss, :] * 2 / (u_max - u_min)
                dN_dv = mesh.C[i_elem] @ dBdv[0, i_gauss, :] * 2 / (v_max - v_min)
            else:
                N_mat = mesh.C[i_elem] @ Buv[i_gauss, 0, :]
                dN_du = mesh.C[i_elem] @ dBdu[i_gauss, 0, :] * 2 / (u_max - u_min)
                dN_dv = mesh.C[i_elem] @ dBdv[i_gauss, 0, :] * 2 / (v_max - v_min)

            # compute the rational basis
            RR = N_mat * wgts
            dRdu = dN_du * wgts
            dRdv = dN_dv * wgts
            w_sum = np.sum(RR)
            dw_xi = np.sum(dRdu)
            dw_eta = np.sum(dRdv)

            dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
            dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2

            # compute the solution w.r.t. the physical space
            dR = np.stack((dRdu, dRdv))
            dxdxi = dR @ cpts.transpose()

            # Jacobian of face mapping
            if side == "right" or side == "left":
                e_jac = dxdxi[1, 0] ** 2 + dxdxi[1, 1] ** 2
            else:
                e_jac = dxdxi[0, 0] ** 2 + dxdxi[0, 1] ** 2

            jac_par_phys = np.sqrt(e_jac)
            dR = np.linalg.solve(dxdxi, dR)
            RR /= w_sum
            phys_pt = cpts @ RR
            g_func = op_val(phys_pt[0], phys_pt[1])
            local_length = (
                jac_par_phys * jac_par_ref * quad_rule_side["weights"][i_gauss]
            )
            local_rhs += RR[sctr] * g_func * local_length
            boundary_length += local_length
        rhs[global_nodes[sctr]] += local_rhs
    print("The boundary length is ", boundary_length)
    return rhs


def applyRobinScalar3D(mesh, lhs, rhs, side, quad_rule_side, alpha, op_val):
    """
    !!!!!!!!!!Not yet converted to 3D!!!!!!!!!!!!!!!
    Applies the Robin boundary conditins on the given patch and side for
    scalar problems and updates the rhs vector

    Parameters
    ----------
    mesh :(IGAMesh2D) patch mesh
    lhs : (2D array) lhs vector
    rhs : (1D array) rhs vector
    side : (string) either "down" (v=0), "right" (u=1), "up" (v=1), "left" (u=0)
    quad_rule_side : (dict) auss points and weights in the reference
                interval [-1,1]
    alpha: (float) coefficient for Robin boundary conditions
    op_val : (function) function for the flux

    Returns
    -------
    rhs : (1D array)
        updated rhs vector

    """
    index_all = mesh._get_boundary_indices()
    num_nodes_edge = len(quad_rule_side["nodes"])
    if side == "right":
        pts_u = np.array([1.0])
        pts_v = quad_rule_side["nodes"]
    elif side == "left":
        pts_u = np.array([-1.0])
        pts_v = quad_rule_side["nodes"]
    elif side == "down":
        pts_u = quad_rule_side["nodes"]
        pts_v = np.array([-1.0])
    elif side == "up":
        pts_u = quad_rule_side["nodes"]
        pts_v = np.array([1.0])
    sctr = index_all[side]
    num_nodes = len(sctr)

    # Allocate memory for the triplet arrays
    index_counter = len(mesh.elem[side]) * len(sctr) ** 2
    II = np.zeros(index_counter, dtype=int)
    JJ = np.zeros(index_counter, dtype=int)
    M = np.zeros(index_counter, dtype=np.complex128)

    index_counter = 0

    # Form the 2D tensor product of the basis functions
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh.deg)

    # Evaluate the Neumann integral on each element
    boundary_length = 0.0
    for i_elem in mesh.elem[side]:
        u_min = mesh.elem_vertex[i_elem, 0]
        u_max = mesh.elem_vertex[i_elem, 2]
        v_min = mesh.elem_vertex[i_elem, 1]
        v_max = mesh.elem_vertex[i_elem, 3]
        if side == "right" or side == "left":
            jac_par_ref = (v_max - v_min) / 2
        else:
            jac_par_ref = (u_max - u_min) / 2

        # compute the rational spline basis
        local_nodes = mesh.elem_node[i_elem]
        global_nodes = mesh.elem_node_global[i_elem]

        cpts = mesh.cpts[0:2, local_nodes]
        wgts = mesh.wgts[local_nodes]
        local_rhs = np.zeros(len(sctr))
        local_bnd_mass = np.zeros((num_nodes_edge, num_nodes_edge), dtype=np.complex128)
        for i_gauss in range(len(quad_rule_side["nodes"])):
            # compute the (B-)spline basis functions and derivatives with Bezier extraction
            if side == "right" or side == "left":
                N_mat = mesh.C[i_elem] @ Buv[0, i_gauss, :]
                dN_du = mesh.C[i_elem] @ dBdu[0, i_gauss, :] * 2 / (u_max - u_min)
                dN_dv = mesh.C[i_elem] @ dBdv[0, i_gauss, :] * 2 / (v_max - v_min)
            else:
                N_mat = mesh.C[i_elem] @ Buv[i_gauss, 0, :]
                dN_du = mesh.C[i_elem] @ dBdu[i_gauss, 0, :] * 2 / (u_max - u_min)
                dN_dv = mesh.C[i_elem] @ dBdv[i_gauss, 0, :] * 2 / (v_max - v_min)

            # compute the rational basis
            RR = N_mat * wgts
            dRdu = dN_du * wgts
            dRdv = dN_dv * wgts
            w_sum = np.sum(RR)
            dw_xi = np.sum(dRdu)
            dw_eta = np.sum(dRdv)

            dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
            dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2

            # compute the solution w.r.t. the physical space
            dR = np.stack((dRdu, dRdv))
            dxdxi = dR @ cpts.transpose()

            # Jacobian of face mapping
            if side == "right" or side == "left":
                e_jac = dxdxi[1, 0] ** 2 + dxdxi[1, 1] ** 2
            else:
                e_jac = dxdxi[0, 0] ** 2 + dxdxi[0, 1] ** 2

            jac_par_phys = np.sqrt(e_jac)
            dR = np.linalg.solve(dxdxi, dR)
            RR /= w_sum
            phys_pt = cpts @ RR
            g_func = op_val(phys_pt[0], phys_pt[1])
            local_length = (
                jac_par_phys * jac_par_ref * quad_rule_side["weights"][i_gauss]
            )
            local_rhs += RR[sctr] * g_func * local_length
            local_bnd_mass += np.outer(RR[sctr], RR[sctr]) * alpha * local_length
            boundary_length += local_length
        rhs[global_nodes[sctr]] += local_rhs
        II[index_counter : index_counter + num_nodes ** 2] = np.tile(
            global_nodes[sctr], num_nodes
        )
        JJ[index_counter : index_counter + num_nodes ** 2] = np.repeat(
            global_nodes[sctr], num_nodes
        )
        M[index_counter : index_counter + num_nodes ** 2] = np.reshape(
            local_bnd_mass, num_nodes ** 2
        )
        index_counter += num_nodes ** 2
    size_basis = lhs.shape[0]
    bnd_mass = sparse.coo_matrix((M, (II, JJ)), shape=(size_basis, size_basis))
    lhs += bnd_mass
    print("The boundary length is ", boundary_length)
    return lhs, rhs


def applyNeumannElast3D(mesh, rhs, side, quad_rule, op_val):
    """
    Applies the Neumann boundary conditions on the given patch and side and
    updates the rhs vector

    Parameters
    ----------
    mesh : (IGAMesh3D) patch mesh
    rhs : (1D array) rhs vector
    side : (string) either "down" (v=0), "right" (u=1), "up" (v=1), "left" (u=0)
    quad_rule : (list of dicts) Gauss points and weights in the reference
                interval [-1,1] for each side
    op_val : (function) function for the x and y component of the traction

    Returns
    -------
     rhs: updated rhs vector

    """
    index_all = mesh._get_boundary_indices()
    if side == "right":
        pts_u = np.array([1.0])
        pts_v = quad_rule[1]["nodes"]
        pts_w = quad_rule[2]["nodes"]
    elif side == "left":
        pts_u = np.array([-1.0])
        pts_v = quad_rule[1]["nodes"]
        pts_w = quad_rule[2]["nodes"]
    elif side == "down":
        pts_u = quad_rule[0]["nodes"]
        pts_v = quad_rule[1]["nodes"]
        pts_w = np.array([-1.0])
    elif side == "up":
        pts_u = quad_rule[0]["nodes"]
        pts_v = quad_rule[1]["nodes"]
        pts_w = np.array([1.0])
    elif side == "front":
        pts_u = quad_rule[0]["nodes"]
        pts_v = np.array([-1.0])
        pts_w = quad_rule[2]["nodes"]
    elif side == "back":
        pts_u = quad_rule[0]["nodes"]
        pts_v = np.array([1.0])
        pts_w = quad_rule[2]["nodes"]
    sctr = index_all[side]
    # print("pts_u = ", pts_u)
    # print("pts_v = ", pts_v)
    # Form the 2D tensor product of the basis functions
    Buv, dBdu, dBdv, dBdw = bernstein_basis_3d(pts_u, pts_v, pts_w, mesh.deg)
 
    # Evaluate the Neumann integral on each element
    boundary_area = 0.
    phys_pts = []
    local_areas = []
    normals = []
    for i_elem in mesh.elem[side]:
        u_min = mesh.elem_vertex[i_elem, 0]
        u_max = mesh.elem_vertex[i_elem, 3]
        v_min = mesh.elem_vertex[i_elem, 1]
        v_max = mesh.elem_vertex[i_elem, 4]
        w_min = mesh.elem_vertex[i_elem, 2]
        w_max = mesh.elem_vertex[i_elem, 5]
        if side == "right" or side == "left":
            jac_par_ref = (v_max - v_min)*(w_max - w_min) / 4
        elif side == "up" or side == "down":
            jac_par_ref = (u_max - u_min)*(v_max - v_min) / 4
        elif side == "front" or side == "back":
            jac_par_ref = (u_max - u_min)*(w_max - w_min) / 4

        # compute the rational spline basis
        local_nodes = mesh.elem_node[i_elem]
        global_nodes = mesh.elem_node_global[i_elem]
        global_nodes_xyz = np.reshape(
            np.stack((3 * global_nodes[sctr], 3 * global_nodes[sctr] + 1, 3 * global_nodes[sctr] + 2), axis=1),
            3 * len(sctr),
        )
        cpts = mesh.cpts[0:3, local_nodes]
        wgts = mesh.wgts[local_nodes]
        local_rhs = np.zeros(3 * len(sctr))
        for i_gauss in range(len(quad_rule[0]["nodes"])):
            for j_gauss in range(len(quad_rule[0]["nodes"])):
                # compute the (B-)spline basis functions and derivatives with Bezier extraction
                if side == "right" or side == "left":
                    N_mat = mesh.C[i_elem] @ Buv[0, i_gauss, j_gauss, :]
                    dN_du = mesh.C[i_elem] @ dBdu[0, i_gauss, j_gauss, :] * 2 / (u_max - u_min)
                    dN_dv = mesh.C[i_elem] @ dBdv[0, i_gauss, j_gauss, :] * 2 / (v_max - v_min)
                    dN_dw = mesh.C[i_elem] @ dBdv[0, i_gauss, j_gauss, :] * 2 / (w_max - w_min)
                elif side == "up" or side == "down":
                    N_mat = mesh.C[i_elem] @ Buv[i_gauss, j_gauss, 0, :]
                    dN_du = mesh.C[i_elem] @ dBdu[i_gauss, j_gauss, 0, :] * 2 / (u_max - u_min)
                    dN_dv = mesh.C[i_elem] @ dBdv[i_gauss, j_gauss, 0, :] * 2 / (v_max - v_min)
                    dN_dw = mesh.C[i_elem] @ dBdw[i_gauss, j_gauss, 0, :] * 2 / (w_max - w_min)                    
                elif side == "front" or side == "back":
                    N_mat = mesh.C[i_elem] @ Buv[i_gauss, 0, j_gauss, :]
                    dN_du = mesh.C[i_elem] @ dBdu[i_gauss, 0, j_gauss, :] * 2 / (u_max - u_min)
                    dN_dv = mesh.C[i_elem] @ dBdv[i_gauss, 0, j_gauss, :] * 2 / (v_max - v_min)
                    dN_dw = mesh.C[i_elem] @ dBdw[i_gauss, 0, j_gauss, :] * 2 / (w_max - w_min)                    

                # compute the rational basis
                RR = N_mat * wgts
                dRdu = dN_du * wgts
                dRdv = dN_dv * wgts
                dRdw = dN_dw * wgts
                w_sum = np.sum(RR)
                dw_xi = np.sum(dRdu)
                dw_eta = np.sum(dRdv)
                dw_zeta = np.sum(dRdw)
    
                dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
                dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2
                dRdw = dRdw / w_sum - RR * dw_zeta / w_sum ** 2
    
                # compute the solution w.r.t. the physical space
                dR = np.stack((dRdu, dRdv, dRdw))
                dxdxi = dR @ cpts.transpose()
                nor = np.zeros(3)
                # Jacobian of face mapping
                if side == "front":
                    nor[0] = dxdxi[0,1]*dxdxi[2,2] - dxdxi[0,2]*dxdxi[2,1]
                    nor[1] = dxdxi[0,2]*dxdxi[2,0] - dxdxi[0,0]*dxdxi[2,2]
                    nor[2] = dxdxi[0,0]*dxdxi[2,1] - dxdxi[0,1]*dxdxi[2,0]
                elif side == "right":
                    nor[0] = dxdxi[1,1]*dxdxi[2,2] - dxdxi[1,2]*dxdxi[2,1]
                    nor[1] = dxdxi[1,2]*dxdxi[2,0] - dxdxi[1,0]*dxdxi[2,2]
                    nor[2] = dxdxi[1,0]*dxdxi[2,1] - dxdxi[1,1]*dxdxi[2,0]
                elif side == "back":
                    nor[0] = -dxdxi[0,1]*dxdxi[2,2] + dxdxi[0,2]*dxdxi[2,1]
                    nor[1] = -dxdxi[0,2]*dxdxi[2,0] + dxdxi[0,0]*dxdxi[2,2]
                    nor[2] = -dxdxi[0,0]*dxdxi[2,1] + dxdxi[0,1]*dxdxi[2,0]
                elif side == "left":
                    nor[0] = -dxdxi[1,1]*dxdxi[2,2] + dxdxi[1,2]*dxdxi[2,1]
                    nor[1] = -dxdxi[1,2]*dxdxi[2,0] + dxdxi[1,0]*dxdxi[2,2]
                    nor[2] = -dxdxi[1,0]*dxdxi[2,1] + dxdxi[1,1]*dxdxi[2,0]
                elif side == "down":
                    nor[0] = dxdxi[1,1]*dxdxi[0,2] - dxdxi[1,2]*dxdxi[0,1]
                    nor[1] = dxdxi[1,2]*dxdxi[0,0] - dxdxi[1,0]*dxdxi[0,2]
                    nor[2] = dxdxi[1,0]*dxdxi[0,1] - dxdxi[1,1]*dxdxi[0,0]
                elif side == "up":
                    nor[0] = -dxdxi[1,1]*dxdxi[0,2] + dxdxi[1,2]*dxdxi[0,1]
                    nor[1] = -dxdxi[1,2]*dxdxi[0,0] + dxdxi[1,0]*dxdxi[0,2]
                    nor[2] = -dxdxi[1,0]*dxdxi[0,1] + dxdxi[1,1]*dxdxi[0,0]
                                             
                jac_par_phys = np.linalg.norm(nor)
                normal = nor / jac_par_phys
                
                
                
                dR = np.linalg.solve(dxdxi, dR)
                RR /= w_sum
                phys_pt = cpts @ RR
                #print("phys_pt =", phys_pt)
                #print("jac_par_ref =", jac_par_ref)
                g_func = op_val(phys_pt[0], phys_pt[1], phys_pt[2], normal[0], normal[1], normal[2])
                # print("g_func =", g_func)
                local_area = (
                    jac_par_phys * jac_par_ref * quad_rule[0]["weights"][i_gauss] * \
                        quad_rule[0]["weights"][j_gauss]
                )
                local_rhs[0:-2:3] += RR[sctr] * g_func[0] * local_area
                local_rhs[1:-1:3] += RR[sctr] * g_func[1] * local_area
                local_rhs[2::3] += RR[sctr] * g_func[2] * local_area
                boundary_area += local_area
                phys_pts.append(phys_pt)
                local_areas.append(local_area)
                normals.append(normal)
        rhs[global_nodes_xyz] += local_rhs
    print("The boundary area is ", boundary_area)
    return rhs, phys_pts, normals


def get_bcdof_bcval(mesh_list, bound_cond):
    '''
    Gets the degree-of-freedom indices and values corresponding to the Dirichlet
    boundary conditions
    TODO: At the moment, only homogeneous Dirichlet B.C. are implemented


    Parameters
    ----------
    mesh_list : (list of IGAMesh3D) multipatch mesh
    bound_cond : (list of boundary3D) boundary conditions


    Returns
    -------
    bcdof : (1d array) the degree of freedom indices
    bcval : (1d array) the values corresponding to each degree of freedom
    '''

    # collect the dofs and values corresponding to the boundary
    bcdof = []
    bcval = []
    eval_pt = [0., 0., 0.]
    for i in range(len(bound_cond)):
        patch_index = bound_cond[i].patch_index
        side = bound_cond[i].side
        if bound_cond[i].type == "Dirichlet":
            # check x-direction
            bound_val = bound_cond[i].op_value(eval_pt[0], eval_pt[1], eval_pt[2])
            if bound_val[0] != None:
                bcdof += [3 * j for j in mesh_list[patch_index].bcdof_global[side]]
                bcval += [bound_val[0] for j in mesh_list[patch_index].bcdof_global[side]]
            # check y-direction
            if bound_val[1] != None:
                bcdof += [3 * j + 1 for j in mesh_list[patch_index].bcdof_global[side]]
                bcval += [bound_val[1] for j in mesh_list[patch_index].bcdof_global[side]]
            # check z-direction
            if bound_val[2] != None:
                bcdof += [3 * j + 2 for j in mesh_list[patch_index].bcdof_global[side]]
                bcval += [bound_val[2] for j in mesh_list[patch_index].bcdof_global[side]]

    bcdof, ind = np.unique(bcdof, return_index=True)
    bcval = np.array(bcval)[ind]
    return bcdof, bcval
    
def applyBCElast3D(mesh_list, bound_cond, lhs, rhs, quad_rule):
    """
    Applies the boundary conditions to a linear system for 3D elasticity
    TODO: At the moment, only Neumann and homogeneous Dirichlet B.C. are implemented


    Parameters
    ----------
    mesh_list : (list of IGAMesh3D) multipatch mesh
    bound_cond : (list of boundary3D) boundary conditions
    lhs : (2D array) stiffness matrix
    rhs : (1D array) rhs vector
    quad_rule : (list of dicts) list of Gauss points and weights in the reference
                interval [-1,1] (one for each parametric direction)

    Returns
    -------
    lhs: updated stiffness matrix
    rhs: updated rhs vector
    """
    for i in range(len(bound_cond)):
        patch_index = bound_cond[i].patch_index
        side = bound_cond[i].side
        if bound_cond[i].type == "Neumann":            
            rhs, phys_pts, local_areas = applyNeumannElast3D(
                mesh_list[patch_index],
                rhs,
                side,
                quad_rule,
                bound_cond[i].op_value,
            )
    
    bcdof, bcval = get_bcdof_bcval(mesh_list, bound_cond)
    rhs = rhs - lhs[:, bcdof] * bcval
    #diag_avg = np.mean(lhs.diagonal())
    rhs[bcdof] = bcval#*diag_avg
    # TODO: fix this warning
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        lhs[bcdof, :] = 0
        lhs[:, bcdof] = 0
        lhs[bcdof, bcdof] = 1.#diag_avg

    return lhs, rhs, phys_pts, local_areas


def applyBCElast_DEM_3D(mesh_list, bound_cond, size_basis, quad_rule, index_map):
    """
    Applies the Neumann boundary conditions to a linear system for 3D elasticity
    
    Parameters
    ----------
    mesh_list : (list of IGAMesh3D) multipatch mesh
    bound_cond : (list of boundary3D) boundary conditions
    size_basis : (int) size of the basis space
    quad_rule : (list of dicts) list of Gauss points and weights in the reference
                interval [-1,1] (one for each parametric direction)

    Returns
    -------
    rhs: the rhs vector that for the ith entry has rhs[i] = \int_\Gamma_N \phi_i * trac_i
    """
    
    rhs = np.zeros(3*size_basis)    
    for i in range(len(bound_cond)):
        patch_index = bound_cond[i].patch_index
        side = bound_cond[i].side
        if bound_cond[i].type == "Neumann":                       
            rhs, _, _ = applyNeumannElast3D(
                mesh_list[patch_index],
                rhs,
                side,
                quad_rule,
                bound_cond[i].op_value,
            )
    return rhs
