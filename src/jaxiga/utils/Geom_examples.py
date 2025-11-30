# -*- coding: utf-8 -*-
"""
Example geometries extending the base class
"""

from jaxiga.utils.Geom import Geometry1D, Geometry2D, Geometry3D
import numpy as np

class Segment(Geometry1D):
    """
    Class for definining a segment domain
    Input: endpoints - array of the form [[x1], [x2]]
               containing the endpoints (control-points)        
    """
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.x1 = endpoints[0][0]
        self.x2 = endpoints[1][0]
        
        geomData = dict()
        
        # set degrees
        geomData["degree_u"] = 1
        geomData["degree_v"] = 1
        
        # Set the control points
        geomData["ctrlpts_size_u"] = 1

        geomData["ctrlpts"] = [
            [self.x1, 0., 0.],
            [self.x2, 0., 0.]            
        ]
        geomData["weights"] = [1.0, 1.0]
        # Set knot vectors
        geomData["knotvector_u"] = [0.0, 0.0, 1.0, 1.0]
        super().__init__(geomData)            


class Quadrilateral(Geometry2D):
    """
    Class for definining a quadrilateral domain
    Input: quadDom - array of the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
               containing the domain corners (control-points), where the points
               (x1, y1), (x3,y3), (x4, y4) and (x2, y2) are in counter-clockwise order.
    """

    def __init__(self, quadDom):

        # Domain vertices
        self.quadDom = quadDom

        self.x1, self.y1 = self.quadDom[0, :]
        self.x2, self.y2 = self.quadDom[1, :]
        self.x3, self.y3 = self.quadDom[2, :]
        self.x4, self.y4 = self.quadDom[3, :]

        geomData = dict()

        # Set the degrees
        geomData["degree_u"] = 1
        geomData["degree_v"] = 1

        # Set the control points
        geomData["ctrlpts_size_u"] = 2
        geomData["ctrlpts_size_v"] = 2

        geomData["ctrlpts"] = [
            [self.x1, self.y1, 0],
            [self.x2, self.y2, 0],
            [self.x3, self.y3, 0],
            [self.x4, self.y4, 0],
        ]

        geomData["weights"] = [1.0, 1.0, 1.0, 1.0]

        # Set knot vectors
        geomData["knotvector_u"] = [0.0, 0.0, 1.0, 1.0]
        geomData["knotvector_v"] = [0.0, 0.0, 1.0, 1.0]
        super().__init__(geomData)


class Disk(Geometry2D):
    """
    Class for defining a disk domain using 9 control points
    Input: center - array of the form [x,y] containing the disk center
           radius - disk radius
    """

    def __init__(self, center, radius):

        # unweighted control points for the unit center
        cptsUnit = [
            [-1.0, 0.0, 0.0],
            [-1, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, -1, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]

        # scale and translate to a circle with given center and radius
        cptsDisk = np.array(cptsUnit) * radius + center

        # weigh the control points
        weights = [
            1.0,
            1 / np.sqrt(2),
            1.0,
            1 / np.sqrt(2),
            1.0,
            1 / np.sqrt(2),
            1.0,
            1 / np.sqrt(2),
            1,
        ]

        for i in range(3):
            for j in range(9):
                cptsDisk[j, i] = cptsDisk[j, i] * weights[j]

        geomData = dict()

        # Set degrees
        geomData["degree_u"] = 2
        geomData["degree_v"] = 2

        # Set control points
        geomData["ctrlpts_size_u"] = 3
        geomData["ctrlpts_size_v"] = 3

        geomData["ctrlpts"] = cptsDisk.tolist()
        geomData["weights"] = weights

        # Set knot vectors
        geomData["knotvector_u"] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        geomData["knotvector_v"] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        super().__init__(geomData)


class QuarterAnnulus(Geometry2D):
    """
    Class for defining a quarater annulus domain
    Input: radius_int - interior radius
           radius_ext - exterior radius

    """

    def __init__(self, radius_int, radius_ext):

        # unweighted control points for the unit circle
        cptsAnnulus = [
            [radius_int, 0.0, 0.0],
            [radius_int, radius_int, 0.0],
            [0.0, radius_int, 0.0],
            [radius_ext, 0.0, 0.0],
            [radius_ext, radius_ext, 0.0],
            [0.0, radius_ext, 0.0],
        ]

        # weigh the control points
        weights = [1.0, 1 / np.sqrt(2), 1.0, 1, 1 / np.sqrt(2), 1.0]
        for i in range(3):
            for j in range(6):
                # print("cptsAnnulus = ",cptsAnnulus)
                cptsAnnulus[j][i] = cptsAnnulus[j][i] * weights[j]

        geomData = dict()

        # Set degrees
        geomData["degree_u"] = 1
        geomData["degree_v"] = 2

        # Set control points
        geomData["ctrlpts_size_u"] = 2
        geomData["ctrlpts_size_v"] = 3

        geomData["ctrlpts"] = cptsAnnulus
        geomData["weights"] = weights

        # Set knot vectors
        geomData["knotvector_u"] = [0.0, 0.0, 1.0, 1.0]
        geomData["knotvector_v"] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        super().__init__(geomData)
        



class PlateWHole(Geometry2D):
    """
    Class for definining a plate with a hole domain in the 2nd quadrant
           (C^0 parametrization with repeated knot in u direction)
    Input: rad_int - radius of the hole
           lenSquare - length of the modeled plate
    """

    def __init__(self, radInt, lenSquare):

        geomData = dict()

        # Set degrees
        geomData["degree_u"] = 2
        geomData["degree_v"] = 1

        # Set control points
        geomData["ctrlpts_size_u"] = 5
        geomData["ctrlpts_size_v"] = 2

        geomData["ctrlpts"] = [
            [-1.0 * radInt, 0.0, 0.0],
            [-1.0 * lenSquare, 0.0, 0.0],
            [-0.853553390593274 * radInt, 0.353553390593274 * radInt, 0.0],
            [-1.0 * lenSquare, 0.5 * lenSquare, 0.0],
            [-0.603553390593274 * radInt, 0.603553390593274 * radInt, 0.0],
            [-1.0 * lenSquare, 1.0 * lenSquare, 0.0],
            [-0.353553390593274 * radInt, 0.853553390593274 * radInt, 0.0],
            [-0.5 * lenSquare, 1.0 * lenSquare, 0.0],
            [0, 1.0 * radInt, 0],
            [0.0, 1.0 * lenSquare, 0.0],
        ]

        geomData["weights"] = [
            1.0,
            1.0,
            0.853553390593274,
            1.0,
            0.853553390593274,
            1.0,
            0.853553390593274,
            1.0,
            1.0,
            1.0,
        ]

        # Set knot vectors
        geomData["knotvector_u"] = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
        geomData["knotvector_v"] = [0.0, 0.0, 1.0, 1.0]

        super().__init__(geomData)


class PlateWHoleQuadrant(Geometry2D):
    """
    Class for definining a plate with a hole domain in the given quadrant
           (C^0 parametrization with repeated knot in u direction)
    Input: rad_int - radius of the hole
           lenSquare - length of the modeled plate
           quadrant - the quadrant for domain
    """

    def __init__(self, radInt, lenSquare, quadrant):
        geomData = dict()

        # Set degrees
        geomData["degree_u"] = 2
        geomData["degree_v"] = 1

        # Set control points
        geomData["ctrlpts_size_u"] = 5
        geomData["ctrlpts_size_v"] = 2

        ctrl_pts_mat = [
            [-1.0 * radInt, 0.0, 0.0],
            [-1.0 * lenSquare, 0.0, 0.0],
            [-0.853553390593274 * radInt, 0.353553390593274 * radInt, 0.0],
            [-1.0 * lenSquare, 0.5 * lenSquare, 0.0],
            [-0.603553390593274 * radInt, 0.603553390593274 * radInt, 0.0],
            [-1.0 * lenSquare, 1.0 * lenSquare, 0.0],
            [-0.353553390593274 * radInt, 0.853553390593274 * radInt, 0.0],
            [-0.5 * lenSquare, 1.0 * lenSquare, 0.0],
            [0, 1.0 * radInt, 0],
            [0.0, 1.0 * lenSquare, 0.0],
        ]

        geomData["weights"] = [
            1.0,
            1.0,
            0.853553390593274,
            1.0,
            0.853553390593274,
            1.0,
            0.853553390593274,
            1.0,
            1.0,
            1.0,
        ]

        # Set knot vectors
        geomData["knotvector_u"] = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
        geomData["knotvector_v"] = [0.0, 0.0, 1.0, 1.0]

        # determine the rotation angle according to
        if quadrant == 1:
            theta = -np.pi / 2
        elif quadrant == 2:
            theta = 0
        elif quadrant == 3:
            theta = np.pi / 2
        elif quadrant == 4:
            theta = np.pi
        else:
            raise Exception("Wrong quadrant given")

        # define the roatation matrix and rotate the control points
        rot_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        ctrl_pts_np = np.array(ctrl_pts_mat)[:, 0:2]
        ctrl_pts_z = np.array(ctrl_pts_mat)[:, 2:3]
        ctrl_pts_xy_rot = ctrl_pts_np @ (rot_mat.transpose())
        ctrl_pts_rot = np.concatenate((ctrl_pts_xy_rot, ctrl_pts_z), axis=1).tolist()

        geomData["ctrlpts"] = ctrl_pts_rot
        super().__init__(geomData)

class Cuboid(Geometry3D):
    '''
     Class for definining a quadrilateral domain
     Input: hexDom - array of the form [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3],
                                        [x4, y4, z4], [x5, y5, z5], [x6, y6, z6],
                                        [x7, y7, z7], [x8, y8, z8]]
                containing the domain corners (control-points)
    '''
    def __init__(self, hexDom):
      
         # Domain bounds
        self.hexDom = hexDom
        
        self.x1, self.y1, self.z1 = self.hexDom[0,:]
        self.x2, self.y2, self.z2 = self.hexDom[1,:]
        self.x3, self.y3, self.z3 = self.hexDom[2,:]
        self.x4, self.y4, self.z4 = self.hexDom[3,:]
        self.x5, self.y5, self.z5 = self.hexDom[4,:]
        self.x6, self.y6, self.z6 = self.hexDom[5,:]
        self.x7, self.y7, self.z7 = self.hexDom[6,:]
        self.x8, self.y8, self.z8 = self.hexDom[7,:]
        
        geomData = dict()
        
        # Set degrees
        geomData['degree_u'] = 1
        geomData['degree_v'] = 1
        geomData['degree_w'] = 1
        
        # Set control points
        geomData['ctrlpts_size_u'] = 2
        geomData['ctrlpts_size_v'] = 2
        geomData['ctrlpts_size_w'] = 2
        
        geomData['ctrlpts'] = np.array([[self.x1, self.y1, self.z1], [self.x2, self.y2, self.z2],
                        [self.x3, self.y3, self.z3], [self.x4, self.y4, self.z4], [self.x5, self.y5, self.z5],
                        [self.x6, self.y6, self.z6], [self.x7, self.y7, self.z7], [self.x8, self.y8, self.z8]])

        geomData['weights'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                
        
        # Set knot vectors
        geomData['knotvector_u'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_v'] = [0.0, 0.0, 1.0, 1.0]
        geomData['knotvector_w'] = [0.0, 0.0, 1.0, 1.0]
        
        super().__init__(geomData)

class HollowSphere(Geometry3D):
    'Geometry of one eight of a hollow sphere with control points collapsed at the pole'
    
    def __init__(self, rad_int, rad_ext):
        geomData = dict()
        geomData['degree_u'] = 2
        geomData['degree_v'] = 2
        geomData['degree_w'] = 1
        
        geomData['weights'] = np.array([1, 1/np.sqrt(2), 1, 
                                          1/np.sqrt(2), 1/2, 1/np.sqrt(2),
                                          1, 1/np.sqrt(2), 1,
                                          1, 1/np.sqrt(2), 1,
                                          1/np.sqrt(2), 1/2, 1/np.sqrt(2),
                                          1, 1/np.sqrt(2), 1])
              
        # ctrlpts = [[rad_int, 0., 0], [rad_int, 0, rad_int], [0, 0, rad_int],
        #             [rad_int, rad_int, 0], [rad_int, rad_int, rad_int], [0, 0, rad_int],
        #             [0, rad_int, 0], [0, rad_int, rad_int], [0, 0, rad_int],
        #             [rad_ext, 0, 0], [rad_ext, 0, rad_ext], [0, 0, rad_ext],
        #             [rad_ext, rad_ext, 0], [rad_ext, rad_ext, rad_ext], [0, 0, rad_ext],
        #             [0, rad_ext, 0], [0, rad_ext, rad_ext], [0, 0, rad_ext]]
        
        ctrlpts = [[rad_ext, 0, 0], [rad_ext, 0, rad_ext], [0, 0, rad_ext],
                   [rad_ext, rad_ext, 0], [rad_ext, rad_ext, rad_ext], [0, 0, rad_ext],
                   [0, rad_ext, 0], [0, rad_ext, rad_ext], [0, 0, rad_ext],
                   [rad_int, 0., 0], [rad_int, 0, rad_int], [0, 0, rad_int],
                    [rad_int, rad_int, 0], [rad_int, rad_int, rad_int], [0, 0, rad_int],
                    [0, rad_int, 0], [0, rad_int, rad_int], [0, 0, rad_int]]

        
        
        # ctrlpts = [[rad_int, 0., 0], [rad_int, rad_int, 0], [0, rad_int, 0],
        #            [rad_int, 0, rad_int], [rad_int, rad_int, rad_int], [0, rad_int, rad_int],
        #            [0, 0, rad_int], [0, 0, rad_int], [0, 0, rad_int],
        #            [rad_ext, 0, 0], [rad_ext, rad_ext, 0], [0, rad_ext, 0],
        #            [rad_ext, 0, rad_ext], [rad_ext, rad_ext, rad_ext], [0, rad_ext, rad_ext],
        #            [0, 0, rad_ext], [0, 0, rad_ext], [0, 0, rad_ext]]
        
        # for i in range(3):
        #     for j in range(len(ctrlpts)):
        #         # print("cptsAnnulus = ",cptsAnnulus)
        #         ctrlpts[j][i] = ctrlpts[j][i] * geomData['weights'][j]
        geomData['ctrlpts'] = ctrlpts
        print('ctrlpts = ', ctrlpts)
        geomData['ctrlpts_size_u'] = 3
        geomData['ctrlpts_size_v'] = 3
        geomData['ctrlpts_size_w'] = 2
        
        geomData['knotvector_u'] = [0., 0., 0., 1., 1., 1.]
        geomData['knotvector_v'] = [0., 0., 0., 1., 1., 1.]
        geomData['knotvector_w'] = [0., 0., 1., 1.]

        super().__init__(geomData)
        
        
        
        
        
        
        
        
        

