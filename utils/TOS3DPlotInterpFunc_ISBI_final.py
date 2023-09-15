# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:24:30 2020

@author: remus
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
# register Axes3D class with matplotlib by importing Axes3D
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
from matplotlib import cm

def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
    '''
    http://members.cbio.mines-paristech.fr/~nvaroquaux/tmp/matplotlib/examples/mplot3d/pathpatch3d_demo.html
    Plots the string 's' on the axes 'ax', with position 'xyz', size 'size',
    and rotation angle 'angle'.  'zdir' gives the axis which is to be treated
    as the third dimension.  usetex is a boolean indicating whether the string
    should be interpreted as latex or not.  Any additional keyword arguments
    are passed on to transform_path.

    Note: zdir affects the interpretation of xyz.
    '''
    x, y, z = xyz
    if zdir == "y":
        xy1, z1 = (x, z), y
    elif zdir == "y":
        xy1, z1 = (y, z), x
    else:
        xy1, z1 = (x, y), z

    text_path = TextPath((0, 0), s, size=size, usetex=usetex)
    trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])

    p1 = PathPatch(trans.transform_path(text_path), **kwargs)
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)

def TOS3DPlotInterp(dataOfPatient, tos_key = 'TOSInterploated', title = None, alignCenters = True, restoreOriSlices = False, vmax = None):  
    interpolate = True
    # restoreOriSlices = True
    points_interp1d_method = 'quadratic'
    tos_interp1d_method = 'linear'
    # ‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’
    ifMergeDataByPatient = False
    
    # Align by image center
    patientVertices = [sliceData['AnalysisFv'].vertices for sliceData in dataOfPatient]
    patientVerticesXMean = np.mean(np.concatenate([vertices[:,0] for vertices in patientVertices]))
    patientVerticesYMean = np.mean(np.concatenate([vertices[:,1] for vertices in patientVertices]))    

    NSlicesInterp = 50
    midLayerLen = sum(dataOfPatient[0]['AnalysisFv'].layerid ==3)
    # xsMatInterp, ysMatInterp, TOSMatInterp = [np.zeros((NSlicesInterp, midLayerLen))] * 4
    xsMatInterp = np.zeros((NSlicesInterp, midLayerLen))
    ysMatInterp = np.zeros((NSlicesInterp, midLayerLen))
    TOSMatInterp = np.zeros((NSlicesInterp, midLayerLen))
    
    # Sort slice by spatial location
    sliceSpatialLocOrder = np.argsort([datum['SequenceInfo'] for datum in dataOfPatient])
    dataOfPatient = [dataOfPatient[idx] for idx in sliceSpatialLocOrder]
    
    NSlicesOri = len(dataOfPatient)
    NSectors = len(set(dataOfPatient[0]['AnalysisFv'].sectorid))
    sectors = []
    # for sectorIdx in range(NSectors): sectors.append({})
    # Update Each Sector's Information using the first slice
    slice0Data = dataOfPatient[0]
    for sectorIdx, sectorId in enumerate(range(1, NSectors+1)):
        sectorInSliceFaces = slice0Data['AnalysisFv'].faces[slice0Data['AnalysisFv'].sectorid==sectorId]
        sectorInSliceVertices = slice0Data['AnalysisFv'].vertices[sectorInSliceFaces-1]
        sectors.append({'leftMostX': sectorInSliceVertices[0,0,0], 'leftMostY':sectorInSliceVertices[0,0,1]})
    
    xsMatOri = np.zeros((NSlicesOri, midLayerLen))
    ysMatOri = np.zeros((NSlicesOri, midLayerLen))
    TOSMatOri = np.zeros((NSlicesOri, midLayerLen))
    
    for sliceIdx, sliceData in enumerate(dataOfPatient):            
        sliceMidLayerFaces = [sliceData['AnalysisFv'].faces[idx] for idx in range(len(sliceData['AnalysisFv'].layerid)) if sliceData['AnalysisFv'].layerid[idx]==3]
        sliceMidLayerVertices = np.concatenate([np.expand_dims(sliceData['AnalysisFv'].vertices[sliceMidLayerFace-1], axis=0) for sliceMidLayerFace in sliceMidLayerFaces])
        sliceMidLayerXs = sliceMidLayerVertices[:,:,0]
        sliceMidLayerYs = sliceMidLayerVertices[:,:,1]
        xsMatOri[sliceIdx, :] = np.mean(sliceMidLayerXs, axis=1)
        ysMatOri[sliceIdx, :] = np.mean(sliceMidLayerYs, axis=1)
        if tos_key in sliceData.keys():
            TOSMatOri[sliceIdx, :] = sliceData[tos_key]#[0, sliceData['AnalysisFv'].layerid == 3]
            hasTOS = True
        else:
            hasTOS = False
            TOSMatOri[sliceIdx, :] = 0
            
    if alignCenters:
        for sliceIdx in range(len(dataOfPatient)):
            sliceXsMean = np.mean(xsMatOri[sliceIdx,:])
            sliceYsMean = np.mean(ysMatOri[sliceIdx,:])
            xsMatOri[sliceIdx,:] = xsMatOri[sliceIdx,:] - sliceXsMean + patientVerticesXMean
            ysMatOri[sliceIdx,:] = ysMatOri[sliceIdx,:] - sliceYsMean + patientVerticesYMean

    sliceSpatialLocsOri = np.array([sliceData['SequenceInfo'] for sliceData in dataOfPatient])
    sliceSpatialLocsMin, sliceSpatialLocsMax = np.min(sliceSpatialLocsOri), np.max(sliceSpatialLocsOri)
    sliceSpatialLocsInterp = np.linspace(sliceSpatialLocsMin, sliceSpatialLocsMax, NSlicesInterp)
    # Restore original locations
    if restoreOriSlices:
        for sliceIdx, sliceLoc in enumerate(sliceSpatialLocsOri):
            closestIdx = np.argmin(np.abs(sliceSpatialLocsInterp - sliceLoc))
            sliceSpatialLocsInterp[closestIdx] = sliceLoc
    
    
    zsMatOri = np.repeat(sliceSpatialLocsOri.reshape(-1, 1), axis=1, repeats=midLayerLen)
    zsMatInterp = np.repeat(sliceSpatialLocsInterp.reshape(-1, 1), axis=1, repeats=midLayerLen)
    # Interploation
    for ringLoc in range(xsMatOri.shape[1]):
        # For each location on the ring, i.e. column of matOri            
        xsColOri = xsMatOri[:, ringLoc]
        ysColOri = ysMatOri[:, ringLoc]            
        TOSColOri = TOSMatOri[:, ringLoc]
        
        interpFuncX = interp1d(sliceSpatialLocsOri, xsColOri, kind=points_interp1d_method)
        interpFuncY = interp1d(sliceSpatialLocsOri, ysColOri, kind=points_interp1d_method)
        interpFuncTOS = interp1d(sliceSpatialLocsOri, TOSColOri, kind=tos_interp1d_method)
        xsMatInterp[:, ringLoc] = interpFuncX(sliceSpatialLocsInterp)
        ysMatInterp[:, ringLoc] = interpFuncY(sliceSpatialLocsInterp)
        TOSMatInterp[:, ringLoc] = interpFuncTOS(sliceSpatialLocsInterp)
            
    
    xsFlat, ysFlat, zsFlat, TOSFlat = [data.flatten() for data in [xsMatInterp,ysMatInterp,zsMatInterp, TOSMatInterp]]
    xsOriFlat, ysOriFlat, zsOriFlat, TOSOriFlat = [data.flatten() for data in [xsMatOri,ysMatOri,zsMatOri, TOSMatOri]]
    
    xsOrder = np.argsort(xsOriFlat)
    xsFlatOrdered = xsOriFlat[xsOrder]
    ysFlatOrdered = ysOriFlat[xsOrder]
    zsFlatOrdered = zsOriFlat[xsOrder]
    
    xsGrid, ysGrid = np.meshgrid(xsFlatOrdered, ysFlatOrdered)
    zsGrid, _ = np.meshgrid(zsFlatOrdered,zsFlatOrdered)                    
    
    # xsFlat, ysFlat, zsFlat, TOSFlat = [data.flatten() for data in [xsMatOri,ysMatOri,zsMatOri, TOSMatOri]]
    fig = plt.figure()
    axe = fig.gca(projection='3d')    
    # axe.plot_surface(xsGrid[::10,::10], ysGrid[::10,::10], (xsGrid[::10,::10]-65)**2+(ysGrid[::10,::10]-65)**2)
    # axe.plot_surface(xsGrid, ysGrid, (xsGrid-65)**2+(ysGrid-65)**2)
    # axe.plot_surface(xsGrid, ysGrid, zsGrid)
    # axe.plot_trisurf(xsFlatOrdered, ysFlatOrdered, zsFlatOrdered)   
    # if hasTOS:
    #     color = TOSFlat
    # else:
    #     color = zsFlat
    
    if interpolate:
        color = TOSFlat if hasTOS else zsFlat
        scatterPlot = axe.scatter(xsFlat, ysFlat, zsFlat, c = color, cmap='jet', zorder = 2, vmax = vmax, vmin = 17)
    else:
        color = TOSOriFlat if hasTOS else zsOriFlat
        scatterPlot = axe.scatter(xsOriFlat, ysOriFlat, zsOriFlat, c = color, cmap='jet', zorder = 2, vmax = vmax, vmin = 17)
    # axe.view_init(elev=30., azim=-10)
    axe.view_init(elev=90., azim=-10)
    axe.set_xlabel('X')
    axe.set_ylabel('Y')
    axe.set_zlabel('Spatial Location')
    axe.set_axis_off()
    # axe.set_title('TOS Surface' + '\n ' + patientID2Show.replace('//', '-') + ('\n (FAKE TOS)' if not hasTOS else ''))
    if title is not None:
        axe.set_title(title)
    # axe.set_zlim(np.min(zsFlat) - 15, np.max(zsFlat)+15)
    plt.colorbar(scatterPlot, ax = axe)
    
    
    # Try Draw surface
    
    
    # Draw Sectors
    # https://matplotlib.org/3.1.0/gallery/mplot3d/text3d.html
    # https://github.com/pyplot-examples/pyplot-3d-wedge/blob/master/wedge.py
    draw_sectors = False
    if draw_sectors:
        centerX = np.mean(xsFlat)
        centerY = np.mean(ysFlat)
        centerZ = np.min(zsFlat) - 10
        patches = []
        lines = []
        for sectorIdx, sector in enumerate(sectors):
            leftMostX = sectors[sectorIdx]['leftMostX']# 
            leftMostY = sectors[sectorIdx]['leftMostY']# - centerY
            rightMostX = sectors[(sectorIdx+1)%NSectors]['leftMostX']# - centerX
            rightMostY = sectors[(sectorIdx+1)%NSectors]['leftMostY']# - centerY
            startAngle = np.arctan((leftMostY - centerY) / (leftMostX - centerX)) * 180 / np.pi
            endAngle = np.arctan((rightMostY- centerY) / (rightMostX- centerX)) * 180 / np.pi
            
            lines.append([(leftMostX,leftMostY),(centerX,centerY)])
            
            # axe.text(leftMostX, leftMostY, centerZ, "red", color='red')
            # sectorNames = ['LAD']*3*2 + ['RCA']*3*2 + ['LCX']*3*2
            # sectorNames = [f'LAD{idx}' for idx in range(1,7)] + [f'RCA{idx}' for idx in range(1,7)] + [f'LCX{idx}' for idx in range(1,7)]
            # sectorNames = sectorNames[::-1]
            text3d(axe, ((leftMostX + rightMostX)/2, (leftMostY+rightMostY)/2, centerZ), f'S{sectorIdx+1}', zdir = 'z', size = 1)
            # text3d(axe, ((leftMostX + rightMostX)/2-1, (leftMostY+rightMostY)/2, centerZ), sectorNames[sectorIdx], zdir = 'z', size = 1)
            
            # wedge = Wedge((centerX, centerY), 10, startAngle, endAngle, color='green', alpha=0.4)
            # axe.add_patch(wedge)
            # art3d.pathpatch_2d_to_3d(wedge, z=centerZ, zdir='z')
            # patches.append(wedge)
        linesC = LineCollection(lines,zorder=1,color='green',lw=3, alpha = 0.4)
        axe.add_collection3d(linesC,zs=centerZ)
        # p = art3d.Patch3DCollection(patches, alpha=0.4, zorder = 1)
        # axe.add_collection3d(p)
    # eye = plt.imread('./eye_plot.gif')
    # eye = (eye - np.min(eye)) / (np.max(eye) - np.min(eye))
    # eyeH, eyeW = eye.shape[:2]
    # fakeImgXs = np.arange(-eyeW/2 + centerX,eyeW/2 + centerX)
    # fakeImgYs = np.arange(-eyeH/2 + centerY,eyeH/2 + centerY)
    # fakeImgXsGrid, fakeImgYsGrid = np.meshgrid(fakeImgXs, fakeImgYs)