#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for IGA mesh classes
"""
import numpy as np
from jaxiga.utils.bernstein import bezier_extraction

class IGAMesh1D:
    """
    Class for a single-patch IGA mesh
    Input
    ------
    patch : Geometry1D object
    """

    def __init__(self, patch):
        # Compute the Bezier extraction operator in each space direction
        knot_u = patch.curve.knotvector
        self.deg = patch.curve.degree
        C_u, num_elem_u = bezier_extraction(knot_u, self.deg)
        self.num_elem = num_elem_u

        # Compute the Bezier extraction operator in 2D
        self.C = [None] * self.num_elem
        temp = np.arange(0, self.num_elem)
        index_matrix = np.reshape(temp, (num_elem_u))
        # index_matrix = np.transpose(temp)        
        for i in range(num_elem_u):
            elem_index = index_matrix[i]
            self.C[elem_index] = C_u[i]
        # Compute the IEN array
        IEN, self.elem_vertex = self.makeIEN(patch)

        # Compute the number of basis functions
        len_u = len(knot_u) - self.deg - 1        
        self.num_basis = len_u

        # Set the (unweighted) control points and weights as arrays
        cpts_temp = np.reshape(patch.curve.ctrlpts, (len_u, 3))
        self.cpts = np.transpose(cpts_temp)
        # self.cpts = np.array(patch.surf.ctrlpts).transpose()
        wgts_temp = np.reshape(patch.curve.weights, (len_u))
        self.wgts = np.reshape(np.transpose(wgts_temp), len_u)
        # self.wgts = np.array(patch.surf.weights)

        # Set the IEN as list of arrays
        self.elem_node = [IEN[i, :] for i in range(self.num_elem)]

    def makeIEN(self, patch):
        """
        Create the IEN (node index to element) array for a given knot vector and p and
        elementVertex array

        Input:
            patch - (Geometry1D) geometry patch

        Output:
            IEN - array where each row corresponds to an element (non-empty
                  knot-span) and it contains the basis function indices with
                  support on the element

            element_vertex - array where each row corresponds to an element
                             and it contains the coordinates of the element
                             corners in the parameters space in the format
                             [u_min, u_max]
        """

        knot_u = patch.curve.knotvector
        num_elem_u = len(np.unique(knot_u)) - 1        
        num_elem = num_elem_u
        deg = np.array(patch.curve.degree)
        num_entries = deg + 1        
        IEN = np.zeros((num_elem, num_entries), dtype=int)
        element_vertex = np.zeros((num_elem, 2))
        element_counter = 0
        
        for i in range(len(knot_u) - 1):
            if (knot_u[i + 1] > knot_u[i]):
                element_vertex[element_counter, :] = [
                    knot_u[i],                    
                    knot_u[i + 1]                    
                ]
                # now we add the nodes from i-p,..., i in the u direction                
                tcount = 0                
                for t1 in range(i - deg, i + 1):
                    IEN[element_counter, tcount] = t1
                    tcount += 1
                element_counter += 1
        assert element_counter == num_elem
        return IEN, element_vertex

    def _get_boundary_indices(self):
        """
        Returns the boundary (left, right) indices for a matrix of
        dimension (p+1)

        Returns
        -------
        bnd_index : dict containing the "left", "right" boundary
                    indices
        """
        bnd_index = {}        
        bnd_index["right"] = [self.deg]        
        bnd_index["left"] = [0]
        return bnd_index

    def classify_boundary(self):
        """
        Classifies the boundary nodes and boundary elements according to the
        side in the parameter space (bcdof_left and elem_left for u=0,
        bcdof_right and elem_right for u=1)
        """     
        bcdof_left = []
        elem_left = []
        bcdof_right = []
        elem_right = []
        bnd_index = self._get_boundary_indices()
        tol_eq = 1e-10
        self.bcdof = {}
        self.elem = {}

        for i_elem in range(self.num_elem):
            if abs(self.elem_vertex[i_elem, 0]) < tol_eq:
                # u_min = 0
                bcdof_left.append(self.elem_node[i_elem][bnd_index["left"]])
                elem_left.append(i_elem)           
            if abs(self.elem_vertex[i_elem, 1] - 1) < tol_eq:
                # u_max = 1
                bcdof_right.append(self.elem_node[i_elem][bnd_index["right"]])
                elem_right.append(i_elem)
            
        self.bcdof["left"] = np.unique(bcdof_left).tolist()
        self.bcdof["right"] = np.unique(bcdof_right).tolist()                
        self.elem["left"] = elem_left
        self.elem["right"] = elem_right

class IGAMesh2D:
    """
    Class for a single-patch IGA mesh
    Input
    ------
    patch : Geometry2D object
    """

    def __init__(self, patch):
        # Compute the Bezier extraction operator in each space direction
        knot_u = patch.surf.knotvector_u
        knot_v = patch.surf.knotvector_v
        self.deg = np.array(patch.surf.degree)
        C_u, num_elem_u = bezier_extraction(knot_u, self.deg[0])
        C_v, num_elem_v = bezier_extraction(knot_v, self.deg[1])
        self.num_elem = num_elem_u * num_elem_v

        # Compute the Bezier extraction operator in 2D
        self.C = [None] * self.num_elem
        temp = np.arange(0, self.num_elem)
        index_matrix = np.reshape(temp, (num_elem_v, num_elem_u))
        # index_matrix = np.transpose(temp)
        for j in range(num_elem_v):
            for i in range(num_elem_u):
                elem_index = index_matrix[j, i]

                self.C[elem_index] = np.kron(C_v[j], C_u[i])
        # Compute the IEN array
        IEN, self.elem_vertex = self.makeIEN(patch)

        # Compute the number of basis functions
        len_u = len(knot_u) - self.deg[0] - 1
        len_v = len(knot_v) - self.deg[1] - 1
        self.num_basis = len_u * len_v

        # Set the (unweighted) control points and weights as arrays
        cpts_temp = np.reshape(patch.surf.ctrlpts, (len_u, len_v, 3))
        self.cpts = np.reshape(np.transpose(cpts_temp, (1, 0, 2)), (len_u * len_v, 3))
        self.cpts = np.transpose(self.cpts)
        # self.cpts = np.array(patch.surf.ctrlpts).transpose()
        wgts_temp = np.reshape(patch.surf.weights, (len_u, len_v))
        self.wgts = np.reshape(np.transpose(wgts_temp), len_u * len_v)
        # self.wgts = np.array(patch.surf.weights)

        # Set the IEN as list of arrays
        self.elem_node = [IEN[i, :] for i in range(self.num_elem)]

    def makeIEN(self, patch):
        """
        Create the IEN (node index to element) array for a given knot vector and p and
        elementVertex array

        Input:
            patch - (Geometry2D) geometry patch

        Output:
            IEN - array where each row corresponds to an element (non-empty
                  knot-span) and it contains the basis function indices with
                  support on the element

            element_vertex - array where each row corresponds to an element
                             and it contains the coordinates of the element
                             corners in the parameters space in the format
                             [u_min, v_min, u_max, v_max]
        """

        knot_u = patch.surf.knotvector_u
        knot_v = patch.surf.knotvector_v
        num_elem_u = len(np.unique(knot_u)) - 1
        num_elem_v = len(np.unique(knot_v)) - 1
        num_elem = num_elem_u * num_elem_v
        deg = np.array(patch.surf.degree)
        num_entries = np.prod(deg + 1)
        len_u = len(knot_u) - deg[0] - 1
        IEN = np.zeros((num_elem, num_entries), dtype=int)
        element_vertex = np.zeros((num_elem, 4))
        element_counter = 0
        for j in range(len(knot_v) - 1):
            for i in range(len(knot_u) - 1):
                if (knot_u[i + 1] > knot_u[i]) and (knot_v[j + 1] > knot_v[j]):
                    element_vertex[element_counter, :] = [
                        knot_u[i],
                        knot_v[j],
                        knot_u[i + 1],
                        knot_v[j + 1],
                    ]
                    # now we add the nodes from i-p,..., i in the u direction
                    # j-q,..., j in the v direction
                    tcount = 0
                    for t2 in range(j - deg[1], j + 1):
                        for t1 in range(i - deg[0], i + 1):
                            IEN[element_counter, tcount] = t1 + t2 * len_u
                            tcount += 1
                    element_counter += 1
        assert element_counter == num_elem
        return IEN, element_vertex

    def _get_boundary_indices(self):
        """
        Returns the boundary (down, right, up, left) indices for a matrix of
        dimension (p+1)*(q+1)

        Returns
        -------
        bnd_index : dict containing the "down", "up", "left", "right" boundary
                    indices
        """
        bnd_index = {}
        bnd_index["down"] = list(range(self.deg[0] + 1))
        bnd_index["right"] = list(
            range(self.deg[0], np.prod(self.deg + 1), self.deg[0] + 1)
        )
        bnd_index["up"] = list(
            range((self.deg[0] + 1) * self.deg[1], np.prod(self.deg + 1))
        )
        bnd_index["left"] = list(
            range(0, 1 + (self.deg[0] + 1) * self.deg[1], self.deg[0] + 1)
        )
        return bnd_index

    def classify_boundary(self):
        """
        Classifies the boundary nodes and boundary elements according to the
        side in the parameter space (i.e. bcdof_down and elem_down for v=0,
        bcdof_up and elem_up for v=1, bcdof_left and elem_left for u=0,
        bcdof_right and elem_right for u=1)
        """
        bcdof_down = []
        elem_down = []
        bcdof_up = []
        elem_up = []
        bcdof_left = []
        elem_left = []
        bcdof_right = []
        elem_right = []
        bnd_index = self._get_boundary_indices()
        tol_eq = 1e-10
        self.bcdof = {}
        self.elem = {}

        for i_elem in range(self.num_elem):
            if abs(self.elem_vertex[i_elem, 0]) < tol_eq:
                # u_min = 0
                bcdof_left.append(self.elem_node[i_elem][bnd_index["left"]])
                elem_left.append(i_elem)
            if abs(self.elem_vertex[i_elem, 1]) < tol_eq:
                # v_min = 0
                bcdof_down.append(self.elem_node[i_elem][bnd_index["down"]])
                elem_down.append(i_elem)
            if abs(self.elem_vertex[i_elem, 2] - 1) < tol_eq:
                # u_max = 1
                bcdof_right.append(self.elem_node[i_elem][bnd_index["right"]])
                elem_right.append(i_elem)
            if abs(self.elem_vertex[i_elem, 3] - 1) < tol_eq:
                # v_max = 1
                bcdof_up.append(self.elem_node[i_elem][bnd_index["up"]])
                elem_up.append(i_elem)
        self.bcdof["down"] = np.unique(bcdof_down).tolist()
        self.bcdof["up"] = np.unique(bcdof_up).tolist()
        self.bcdof["left"] = np.unique(bcdof_left).tolist()
        self.bcdof["right"] = np.unique(bcdof_right).tolist()
        self.bcdof["down_left"] = np.intersect1d(bcdof_down, bcdof_left).tolist()
        self.bcdof["down_right"] = np.intersect1d(bcdof_down, bcdof_right).tolist()
        self.bcdof["up_left"] = np.intersect1d(bcdof_up, bcdof_left).tolist()
        self.bcdof["up_right"] = np.intersect1d(bcdof_up, bcdof_right).tolist()
        self.elem["down"] = elem_down
        self.elem["up"] = elem_up
        self.elem["left"] = elem_left
        self.elem["right"] = elem_right

class IGAMesh3D:
    """
    Class for a single-patch IGA mesh
    Input
    ------
    patch : Geometry3D object
    """

    def __init__(self, patch):
        # Compute the Bezier extraction operator in each space direction
        self.knot_u = patch.vol.knotvector_u
        self.knot_v = patch.vol.knotvector_v
        self.knot_w = patch.vol.knotvector_w
        self.deg = np.array(patch.vol.degree)
                
        C_u, num_elem_u = bezier_extraction(self.knot_u, self.deg[0])
        C_v, num_elem_v = bezier_extraction(self.knot_v, self.deg[1])
        C_w, num_elem_w = bezier_extraction(self.knot_w, self.deg[2])
        self.num_elem = num_elem_u * num_elem_v * num_elem_w

        # Compute the Bezier extraction operator in 2D
        self.C = [None] * self.num_elem
        elem_counter = 0
        for k in range(num_elem_w):
            for j in range(num_elem_v):
                for i in range(num_elem_u):            
                    self.C[elem_counter] = np.kron(np.kron(C_w[k], C_v[j]), C_u[i])
                    elem_counter += 1
        # Compute the IEN array
        IEN, self.elem_vertex = self.makeIEN()

        # Compute the number of basis functions
        len_u = len(self.knot_u) - self.deg[0] - 1
        len_v = len(self.knot_v) - self.deg[1] - 1
        len_w = len(self.knot_w) - self.deg[2] - 1
        self.num_basis = len_u * len_v * len_w
        
        self.cpts = np.array(patch.vol.ctrlpts).T
        self.wgts = np.array(patch.vol.weights)
        
        # Set the IEN as list of arrays
        self.elem_node = [IEN[i, :] for i in range(self.num_elem)]

    def makeIEN(self):
        """
        Create the IEN (node index to element) array 

        Input:
            none

        Output:
            IEN - array where each row corresponds to an element (non-empty
                  knot-span) and it contains the basis function indices with
                  support on the element

            element_vertex - array where each row corresponds to an element
                             and it contains the coordinates of the element
                             corners in the parameters space in the format
                             [u_min, v_min, w_min, u_max, v_max, w_max]
        """

        num_elem_u = len(np.unique(self.knot_u)) - 1
        num_elem_v = len(np.unique(self.knot_v)) - 1
        num_elem_w = len(np.unique(self.knot_w)) - 1
        num_elem = num_elem_u * num_elem_v * num_elem_w
        num_entries = np.prod(self.deg + 1)
        len_u = len(self.knot_u) - self.deg[0] - 1
        len_v = len(self.knot_v) - self.deg[1] - 1
        IEN = np.zeros((num_elem, num_entries), dtype=int)
        element_vertex = np.zeros((num_elem, 6))
        element_counter = 0
        for k in range(len(self.knot_w) - 1):
            for j in range(len(self.knot_v) - 1):
                for i in range(len(self.knot_u) - 1):
                    if (self.knot_u[i + 1] > self.knot_u[i]) \
                        and (self.knot_v[j + 1] > self.knot_v[j]) \
                            and (self.knot_w[k+1] > self.knot_w[k]):
                                element_vertex[element_counter, :] = [
                                    self.knot_u[i],
                                    self.knot_v[j],
                                    self.knot_w[k],
                                    self.knot_u[i + 1],
                                    self.knot_v[j + 1],
                                    self.knot_w[k + 1]
                                ]
                                # now we add the nodes from i-deg[0],..., i in the u direction
                                # j-deg[1],..., j in the v direction and k-deg[2],...,k in 
                                # the w direction
                                tcount = 0
                                for t3 in range(k-self.deg[2], k+1):
                                    for t2 in range(j - self.deg[1], j + 1):
                                        for t1 in range(i - self.deg[0], i + 1):
                                            IEN[element_counter, tcount] = t1 + \
                                                t2*len_u+t3*len_u*len_v;
                                            tcount += 1
                                element_counter += 1
        assert element_counter == num_elem
        return IEN, element_vertex
    
    def _get_boundary_indices(self):
        """
        Returns the boundary (front, back, down, right, up, left) indices for a matrix of
        dimension (p+1)*(q+1)*(r+1)

        Returns
        -------
        bnd_index : dict containing the "front", "back", "down", "up", "left", "right" boundary
                    indices
        """
        def index(i, j, k):
            return i + (p + 1) * j + (p + 1) * (q + 1) * k
        p = self.deg[0]
        q = self.deg[1]
        r = self.deg[2]
        
        bnd_index = {
            "down": [index(i, j, 0) for i in range(p + 1) for j in range(q + 1)],
            "up": [index(i, j, r) for i in range(p + 1) for j in range(q + 1)],
            "front": [index(i, 0, k) for i in range(p + 1) for k in range(r + 1)],
            "back": [index(i, q, k) for i in range(p + 1) for k in range(r + 1)],
            "left": [index(0, j, k) for j in range(q + 1) for k in range(r + 1)],
            "right": [index(p, j, k) for j in range(q + 1) for k in range(r + 1)]
        }

        return bnd_index
                

    def classify_boundary(self):
        """
        Classifies the boundary nodes and boundary elements according to the
        side in the parameter space (i.e. bcdof_down and elem_down for w=0,
        bcdof_up and elem_up for w=1, bcdof_left and elem_left for u=0,
        bcdof_right and elem_right for u=1, bcdof_front and elem_front for v=0,
        bcdof_back and elem_back for v==1)
        """
        bcdof_down = []
        elem_down = []
        bcdof_up = []
        elem_up = []
        bcdof_left = []
        elem_left = []
        bcdof_right = []
        elem_right = []
        bcdof_front = []
        elem_front = []
        bcdof_back = []
        elem_back = []
        bnd_index = self._get_boundary_indices()
        tol_eq = 1e-10
        self.bcdof = {}
        self.elem = {}

        for i_elem in range(self.num_elem):
            if abs(self.elem_vertex[i_elem, 0]) < tol_eq:
                # u_min = 0
                bcdof_left.append(self.elem_node[i_elem][bnd_index["left"]])
                elem_left.append(i_elem)
            if abs(self.elem_vertex[i_elem, 1]) < tol_eq:
                # v_min = 0
                bcdof_front.append(self.elem_node[i_elem][bnd_index["front"]])
                elem_front.append(i_elem)                                
            if abs(self.elem_vertex[i_elem, 2]) < tol_eq:
                # w_min = 0
                bcdof_down.append(self.elem_node[i_elem][bnd_index["down"]])
                elem_down.append(i_elem)                
            if abs(self.elem_vertex[i_elem, 3] - 1) < tol_eq:
                # u_max = 1
                bcdof_right.append(self.elem_node[i_elem][bnd_index["right"]])
                elem_right.append(i_elem)
            if abs(self.elem_vertex[i_elem, 4] - 1) < tol_eq:
                # v_max = 1
                bcdof_back.append(self.elem_node[i_elem][bnd_index["back"]])
                elem_back.append(i_elem)                
            if abs(self.elem_vertex[i_elem, 5] - 1) < tol_eq:
                # w_max = 1
                bcdof_up.append(self.elem_node[i_elem][bnd_index["up"]])
                elem_up.append(i_elem)
                
        self.bcdof["down"] = np.unique(bcdof_down).tolist()
        self.bcdof["up"] = np.unique(bcdof_up).tolist()
        self.bcdof["left"] = np.unique(bcdof_left).tolist()
        self.bcdof["right"] = np.unique(bcdof_right).tolist()
        self.bcdof["front"] = np.unique(bcdof_front).tolist()
        self.bcdof["back"] = np.unique(bcdof_back).tolist()
        # self.bcdof["down_left"] = np.intersect1d(bcdof_down, bcdof_left).tolist()
        # self.bcdof["down_right"] = np.intersect1d(bcdof_down, bcdof_right).tolist()
        # self.bcdof["up_left"] = np.intersect1d(bcdof_up, bcdof_left).tolist()
        # self.bcdof["up_right"] = np.intersect1d(bcdof_up, bcdof_right).tolist()
        self.elem["down"] = elem_down
        self.elem["up"] = elem_up
        self.elem["left"] = elem_left
        self.elem["right"] = elem_right
        self.elem["front"] = elem_front
        self.elem["back"] = elem_back
        
