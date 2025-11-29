#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for routines related to Bezier extraction
"""
import time
import numpy as np
import jax.numpy as jnp


def bezier_extraction(knot, deg):
    """
    Bezier extraction
    Based on Algorithm 1, from Borden - Isogeometric finite element data
    structures based on Bezier extraction
    """
    m = len(knot) - deg - 1
    a = deg + 1
    b = a + 1
    # Initialize C with the number of non-zero knotspans in the 3rd dimension
    # nb_final = len(np.unique(knot))-1
    C = []
    nb = 1
    C.append(np.eye(deg + 1))
    while b <= m:
        C.append(np.eye(deg + 1))
        i = b
        while (b <= m) and (knot[b] == knot[b - 1]):
            b = b + 1
        multiplicity = b - i + 1
        alphas = np.zeros(deg - multiplicity)
        if multiplicity < deg:
            numerator = knot[b - 1] - knot[a - 1]
            for j in range(deg, multiplicity, -1):
                alphas[j - multiplicity - 1] = numerator / (
                    knot[a + j - 1] - knot[a - 1]
                )
            r = deg - multiplicity
            for j in range(1, r + 1):
                save = r - j + 1
                s = multiplicity + j
                for k in range(deg + 1, s, -1):
                    alpha = alphas[k - s - 1]
                    C[nb - 1][:, k - 1] = (
                        alpha * C[nb - 1][:, k - 1] + (1 - alpha) * C[nb - 1][:, k - 2]
                    )
                if b <= m:
                    C[nb][save - 1 : save + j, save - 1] = C[nb - 1][
                        deg - j : deg + 1, deg
                    ]
            nb = nb + 1
            if b <= m:
                a = b
                b = b + 1
        elif multiplicity == deg:
            if b <= m:
                nb = nb + 1
                a = b
                b = b + 1
    # assert(nb==nb_final)

    return C, nb


def bernstein_basis(uhat, deg):
    """
    Algorithm A1.3 in Piegl & Tiller
    uhat is a 1D array
    Computes the Bernstein polynomial of degree deg at points in uhat

    Parameters
    ----------
    uhat: list of evaluation points on the reference interval [-1, 1]
    deg: degree of Bernstein polynomial (integer)

    Returns
    -------
    B: matrix of size num_pts x (deg+1) containing the deg+1 Bernstein polynomials
       evaluated at the points uhat
    """
    B = jnp.ones((len(uhat), 1))
    u1 = 1 - jnp.expand_dims(uhat, 1)
    u2 = 1 + jnp.expand_dims(uhat, 1)

    for j in range(1, deg + 1):
        B_new = u1*B[:,0:1]
        saved = u2*B[:,0:1]
        for k in range(1, j):
            B_new = jnp.concatenate((B_new, saved + u1 * B[:,k:k+1]), axis=1)
            saved = u2 * B[:, k:k+1]
        B = jnp.concatenate((B_new, saved), axis=1)
    B = B / np.power(2, deg)

    return B


def bernstein_basis_deriv(uhat, deg):
    """
    Algorithm A1.3 in Piegl & Tiller
    uhat is a 1D array
    Computes the derivatives of the Bernstein polynomial of degree deg at point
    uhat on the interval [-1, 1]

    Parameters
    ----------
    uhat: list of evaluation points on the reference interval [-1, 1]
    deg: degree of Bernstein polynomial (integer)

    Returns
    -------
    B: matrix of size num_pts x (deg+1) containing the deg+1 Bernstein polynomials
       evaluated at the points uhat
    """
    dB = jnp.ones((len(uhat), 1))
    u1 = 1 - jnp.expand_dims(uhat, 1)
    u2 = 1 + jnp.expand_dims(uhat, 1)
    for j in range(1, deg):
        dB_new = u1*dB[:,0:1]
        saved = u2*dB[:,0:1]
        for k in range(1, j):
            dB_new = jnp.concatenate((dB_new, saved + u1 * dB[:, k:k+1]), axis=1)
            saved = u2 * dB[:, k:k+1]
        dB = jnp.concatenate((dB_new, saved), axis=1)
    dB = dB / np.power(2, deg)
    dB0 = jnp.transpose(jnp.array([jnp.zeros(len(uhat))]))
    dB = jnp.concatenate((dB0, dB, dB0), axis=1)
    dB = (dB[:, 0:-1] - dB[:, 1:]) * deg

    return dB

def bernstein_basis_2nd_deriv(uhat, deg):
    """
    Calculates the 2nd derivative of the Bernstein basis

    Args:
        uhat (list of floats): list of evaluation points on the reference interval [-1, 1]
        deg (integer): degree of the Bernstein polynomial
    """
    ddB = jnp.ones((len(uhat), 1))
    u1 = 1 - jnp.expand_dims(uhat, 1)
    u2 = 1 + jnp.expand_dims(uhat, 1)
    for j in range(1, deg - 1):
        ddB_new = u1*ddB[:,0:1]
        saved = u2*ddB[:,0:1]
        for k in range(1, j):
            ddB_new = jnp.concatenate((ddB_new, saved + u1 * ddB[:, k:k+1]), axis=1)
            saved = u2 * ddB[:, k:k+1]
        ddB = jnp.concatenate((ddB_new, saved), axis=1)
    ddB = ddB / np.power(2, deg)
    ddB0 = jnp.transpose(jnp.array([jnp.zeros(len(uhat))]))
    ddB = jnp.concatenate((ddB0, ddB0, ddB, ddB0, ddB0), axis=1)
    ddB = (ddB[:, 0:-2] - 2*ddB[:, 1:-1] + ddB[:, 2:]) * deg * (deg - 1)

    return ddB


def form_extended_knot(localKnot, deg):
    """
    Create the extended knot vector (Subsection 4.3.2 in Scott - Isogeometric
    data structures based on the Bézier extraction of T-Splines)

    Parameters
    ----------
    localKnots: the local knot vector (list of length deg+2)
    deg: polynomial degree of the basis

    Returns
    -------
    extendedKnot: the extended knot vector

    """
    # Repeat the first knot (if needed) so that it appears p+1 times
    firstKnot = localKnot[0]
    tol_comp = 1e-12
    indexFirst = [i for i in localKnot if abs(i - firstKnot) < tol_comp]
    numRep = len(indexFirst)
    numNewRepFirst = deg + 1 - numRep

    # repeat the last knot (if needed) so that it appears p+1 times
    lastKnot = localKnot[-1]
    indexLast = [i for i in localKnot if abs(i - lastKnot) < tol_comp]
    numRep = len(indexLast)
    numNewRepLast = deg + 1 - numRep

    # form the extended knot vector
    extendedKnot = np.concatenate(
        (
            firstKnot * np.ones(numNewRepFirst),
            localKnot,
            lastKnot * np.ones(numNewRepLast),
        )
    )
    indexFun = numNewRepFirst
    return extendedKnot, indexFun + 1

def bernstein_basis_1d(pts_u, deg):
    """
    Generates the 1D Bernstein polynomials at points (pts_u)

    Parameters
    ----------
    pts_u : list of evaluation points in the u direction    
    deg (int): polynomial degree

    Returns
    -------
    Bu : values of the basis functions
    dBdu : values of the derivatives of the basis functions with respect to u    

    """
    B_u = bernstein_basis(pts_u, deg)
    dB_u = bernstein_basis_deriv(pts_u, deg)
    
    num_pts_u = len(pts_u)
    num_basis = deg + 1
    basis_counter = 0
    Bu = np.zeros((num_pts_u, num_basis))
    dBdu = np.zeros((num_pts_u, num_basis))
    
    for i in range(deg + 1):
        Bu[:, basis_counter] = B_u[:, i]
        dBdu[:, basis_counter] = dB_u[:, i]        
        basis_counter += 1

    return Bu, dBdu


def bernstein_basis_2d(pts_u, pts_v, deg):
    """
    Generates the 2D Bernstein polynomials at points (pts_u, pts_v)

    Parameters
    ----------
    pts_u : list of evaluation points in the u direction
    pts_v : list of evaluation points in the v direction
    deg : list of polynomial degrees

    Returns
    -------
    Buv : values of the basis functions
    dBdu : values of the derivatives of the basis functions with respect to u
    dBdv : values of the derivatives of the basis functions with respect to v

    """
    B_u = bernstein_basis(pts_u, deg[0])
    B_v = bernstein_basis(pts_v, deg[1])
    dB_u = bernstein_basis_deriv(pts_u, deg[0])
    dB_v = bernstein_basis_deriv(pts_v, deg[1])
    
    num_pts_u = len(pts_u)
    num_pts_v = len(pts_v)
    num_basis = (deg[0] + 1) * (deg[1] + 1)
    basis_counter = 0
    Buv = np.zeros((num_pts_u, num_pts_v, num_basis))
    dBdu = np.zeros((num_pts_u, num_pts_v, num_basis))
    dBdv = np.zeros((num_pts_u, num_pts_v, num_basis))

    for j in range(deg[1] + 1):
        for i in range(deg[0] + 1):
            Buv[:, :, basis_counter] = np.outer(B_u[:, i], B_v[:, j])
            dBdu[:, :, basis_counter] = np.outer(dB_u[:, i], B_v[:, j])
            dBdv[:, :, basis_counter] = np.outer(B_u[:, i], dB_v[:, j])
            basis_counter += 1

    return Buv, dBdu, dBdv

def bernstein_basis_2d_2nd_deriv(pts_u, pts_v, deg):
    """
    Generates the 2D Bernstein polynomials at points (pts_u, pts_v)

    Parameters
    ----------
    pts_u : list of evaluation points in the u direction
    pts_v : list of evaluation points in the v direction
    deg : list of polynomial degrees

    Returns
    -------
    Buv : values of the basis functions
    dBdu : values of the derivatives of the basis functions with respect to u
    dBdv : values of the derivatives of the basis functions with respect to v

    """
    B_u = bernstein_basis(pts_u, deg[0])
    B_v = bernstein_basis(pts_v, deg[1])
    dB_u = bernstein_basis_deriv(pts_u, deg[0])
    dB_v = bernstein_basis_deriv(pts_v, deg[1])
    ddB_u = bernstein_basis_2nd_deriv(pts_u, deg[0])
    ddB_v = bernstein_basis_2nd_deriv(pts_v, deg[1])
    
    num_pts_u = len(pts_u)
    num_pts_v = len(pts_v)
    num_basis = (deg[0] + 1) * (deg[1] + 1)
    basis_counter = 0
    Buv = np.zeros((num_pts_u, num_pts_v, num_basis))
    dBdu = np.zeros((num_pts_u, num_pts_v, num_basis))
    dBdv = np.zeros((num_pts_u, num_pts_v, num_basis))
    ddBv = np.zeros((num_pts_u, num_pts_v, num_basis))
    ddBu = np.zeros((num_pts_u, num_pts_v, num_basis))
    ddBuv = np.zeros((num_pts_u, num_pts_v, num_basis))

    for j in range(deg[1] + 1):
        for i in range(deg[0] + 1):
            Buv[:, :, basis_counter] = np.outer(B_u[:, i], B_v[:, j])
            dBdu[:, :, basis_counter] = np.outer(dB_u[:, i], B_v[:, j])
            dBdv[:, :, basis_counter] = np.outer(B_u[:, i], dB_v[:, j])
            ddBu[:,:,basis_counter] = np.outer(ddB_u[:,i], B_v[:,j])
            ddBv[:,:,basis_counter] = np.outer(B_u[:,i], ddB_v[:,j])
            ddBuv[:,:,basis_counter] = np.outer(dB_u[:,i], dB_v[:,j]);
            basis_counter += 1

    return Buv, dBdu, dBdv

def bernstein_basis_3d(pts_u, pts_v, pts_w, deg):
    """
    Generates the 2D Bernstein polynomials at points (pts_u, pts_v, pts_w)

    Parameters
    ----------
    pts_u : list of evaluation points in the u direction
    pts_v : list of evaluation points in the v direction
    pts_w : list of evaluation points in the w direction
    deg : list of polynomial degrees

    Returns
    -------
    Buv : values of the basis functions
    dBdu : values of the derivatives of the basis functions with respect to u
    dBdv : values of the derivatives of the basis functions with respect to v
    dBdw : values of the derivatives of the basis functions with respect to w

    """
    t1 = time.time()
    B_u = bernstein_basis(pts_u, deg[0])
    B_v = bernstein_basis(pts_v, deg[1])
    B_w = bernstein_basis(pts_w, deg[2])
    t2 = time.time()
    print("Time for evaluating 1D Bezier polynomials is", t2-t1)
    dB_u = bernstein_basis_deriv(pts_u, deg[0])
    dB_v = bernstein_basis_deriv(pts_v, deg[1])
    dB_w = bernstein_basis_deriv(pts_w, deg[2])
    t3 = time.time()
    print("Time for evaluating 1D Bezier polynomial derivatives is", t3-t2)
        
    
    num_pts_u = len(pts_u)
    num_pts_v = len(pts_v)
    num_pts_w = len(pts_w)

    # Calculate the total number of basis functions            
    num_basis = (deg[0] + 1) * (deg[1] + 1) * (deg[2] + 1)
    basis_counter = 0
    Buvw = np.zeros((num_pts_u, num_pts_v, num_pts_w, num_basis))
    dBdu = np.zeros((num_pts_u, num_pts_v, num_pts_w, num_basis))
    dBdv = np.zeros((num_pts_u, num_pts_v, num_pts_w, num_basis))
    dBdw = np.zeros((num_pts_u, num_pts_v, num_pts_w, num_basis))

    for k in range(deg[2] + 1):
        for j in range(deg[1] + 1):
            for i in range(deg[0] + 1):                                    
                Buvw[:, :, :, basis_counter] = np.einsum('i,j,k->ijk', B_u[:, i], B_v[:, j], B_w[:, k])
                dBdu[:, :, :, basis_counter] = np.einsum('i,j,k->ijk', dB_u[:, i], B_v[:, j], B_w[:, k])
                dBdv[:, :, :, basis_counter] = np.einsum('i,j,k->ijk', B_u[:, i], dB_v[:, j], B_w[:, k])
                dBdw[:, :, :, basis_counter] = np.einsum('i,j,k->ijk', B_u[:, i], B_v[:, j], dB_w[:, k])
                basis_counter += 1    

    return Buvw, dBdu, dBdv, dBdw