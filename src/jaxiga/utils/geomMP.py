# -*- coding: utf-8 -*-
"""
Test script for loading a plotting the plate with 3 hole domain with Geomdl library
Import data from nurbsList.mat, where each patch is stored in a cell(1,numPatches) array
"""
import numpy as np
from geomdl import NURBS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from jaxiga.utils.Geom import Geometry2D
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
 

    
def convert_nurbs_to_patches(nurbsList):
    
    patches = []
    numPatches = nurbsList.shape[1]
    for indexPatch in range(numPatches):
        geomData = {}
        nurbsCur = nurbsList[0,indexPatch][0,0]        
        
        # get the degree in each direction (=order-1)
        geomData['degree_u'] = nurbsCur[5][0][0]-1
        geomData['degree_v'] = nurbsCur[5][0][1]-1
        
        # get the number of control points in each direction
        nU = nurbsCur[2][0][0]
        nV = nurbsCur[2][0][1]
        geomData['ctrlpts_size_u'] = int(nU)
        geomData['ctrlpts_size_v'] = int(nV)
        
        # rearrange the control points in the order nUxnVx4 required for geomdl
        coefs = nurbsCur[3]
        geomData['ctrlpts2d'] = np.transpose(coefs,axes=[1,2,0]).tolist()
        
        # get the knot vectors
        geomData['knotvector_u'] = nurbsCur[4][0][0][0].tolist()
        geomData['knotvector_v'] = nurbsCur[4][0][1][0].tolist()
        patches.append(Geometry2D(geomData))
    return patches
            
