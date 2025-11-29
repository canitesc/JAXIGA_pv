#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for material properties
"""
import numpy as np


class MaterialElast2D:
    """
    Class for 2D linear elastic materials
    Input
    -----
    Emod : (float) Young's modulus
    nu : (float) Poisson ratio
    plane_type : (string) stress or strain

    """

    def __init__(self, Emod=None, nu=None, plane_type="stress"):
        self.Emod = Emod
        self.nu = nu
        if plane_type == "stress":
            self.Cmat = (
                Emod
                / (1 - nu ** 2)
                * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
            )
            self.lam = Emod*nu/((1+nu)*(1-2*nu))
            self.mu = Emod/(2*(1+nu))
        elif plane_type == "strain":
            self.Cmat = (
                Emod
                / ((1 + nu) * (1 - 2 * nu))
                * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])
            )


class MaterialElast3D:
    """
    Class for 3D linear elastic materials
    Input
    -----
    Emod : (float) Young's modulus
    nu : (float) Poisson ratio
    """

    def __init__(self, Emod=None, nu=None):
        self.Emod = Emod
        self.nu = nu
        self.Cmat = Emod/(1+nu)/(1-2*nu)*np.array([[1-nu, nu, nu, 0, 0, 0], 
                                                   [nu, 1-nu, nu, 0, 0, 0],
                                                   [nu, nu, 1-nu, 0, 0, 0],
                                                   [0, 0, 0, (1-2*nu)/2, 0, 0],
                                                   [0, 0, 0, 0, (1-2*nu)/2, 0],
                                                   [0, 0, 0, 0, 0, (1-2*nu)/2]])
        
        
        
        