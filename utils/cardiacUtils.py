# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:01:44 2020

@author: remus
"""


import numpy as np
def mat2recvfv(faces, vertices, mat):
    # Fill mat values into faces
    # faces: [Nf,d] ndarray, where d is the number of vertives of each face
    # vertives: [Nv, 2] ndarray
    # mat: [H,W] ndarray
    vals = np.ones(faces.shape[0])*np.nan
    for faceIdx in range(faces.shape[0]):
        verticesOfFace = vertices[faces[faceIdx, :]-1,:]
        xMin = np.min(verticesOfFace[:,0])
        xMax = np.max(verticesOfFace[:,0])
        yMin = np.min(verticesOfFace[:,1])
        yMax = np.max(verticesOfFace[:,1])
        if xMax - xMin > 1:
            x = int(np.round((xMax+xMin)/2))
        else:
            x = int(np.round(xMin))
        if yMax - yMin > 1:
            y = int(np.round((yMax+yMin)/2))
        else:
            y = int(np.round(yMin))
        try:
            vals[faceIdx] = mat[y, x]
        except:
            print(verticesOfFace)
            print(y, x)
    return vals


def rectfv2TOS(vals, sectorids, layerids):
    diffSectorids = np.unique(sectorids)
    NSectors = len(diffSectorids)
    TOS = np.zeros(NSectors)
    for sectorIdx, sectorid in enumerate(diffSectorids):
        valsInSector = vals[(sectorids==sectorid) * (layerids == 3)]
        TOS[sectorIdx] = np.mean(valsInSector)
    return TOS

def getPatientName(filename):
    # Example filename: '../Dataset/CRT_TOS_Data_Jerry/SET03\\UP36/mat/SL4.mat'
    # filename = filename.split('Jerry')[1]
    # filename = filename.split('mat')[0]
    # filename = filename.replace('\\', '-')         # remove '\\'
    # filename = filename.replace('//', '-')         # remove '//'
    # filename = filename.replace('/', '-')          # remove '/'
    # filename = filename.replace('/', '-')          # remove '/'
    # return filename
    return filename.split('Jerry/')[1].split('/mat')[0]

def getSliceName(filename):
    # Example filename: '../Dataset/CRT_TOS_Data_Jerry/SET03\\UP36/mat/SL4.mat'
    filename = filename.split('Jerry/')[1]         # Get the part after 'Jerry/'
    filename = filename.replace('_processed', '')
    filename = filename.replace('.mat', '')        # remove '.mat'
    filename = filename.replace('mat', '')         # remove 'mat'
    filename = filename.replace('\\', '-')         # remove '\\'
    filename = filename.replace('//', '-')         # remove '//'
    filename = filename.replace('/', '-')          # remove '/'
    filename = filename.replace('/', '-')          # remove '/'
    # Exaple final filename: 'SET03-UP36-SL4'
    return filename

def changeNFrames(dataDict: dict, NFramesNew, dataType:str = 'strainMat'):
    # Change number of frames by zeros padding of clip
    data = dataDict[dataType]
    if dataType in ['strainMat', 'strainMatSVD', 'strainCurve','dispFieldJacoMat', 'strainMatFullResolution']:
        # Shape should be [1, 1, NSectors, NFrames] (organized by slice)
        #              or [1, NSLices, NSectorsm NFrames] (organized by patient)
        try:
            N, NSlices, NSectors, NFramesOri = data.shape
        except:
            print(data.shape)
        dataPadded = np.zeros((N, NSlices, NSectors, NFramesNew))
        if NFramesOri < NFramesNew:
            dataPadded[:,:,:,:NFramesOri] = data
        elif NFramesOri > NFramesNew:
            dataPadded = data[:,:,:,:NFramesNew]
        else:
            dataPadded = data
        mask = np.zeros(dataPadded.shape)
        mask[:,:,:,:min(NFramesOri, NFramesNew)] = 1
    elif dataType in ['dispField', 'dispFieldFv']:
        # Shape should be [N, 2, NFrames, H, W]
        N, C, NFramesOri, H, W = data.shape
        dataPadded = np.zeros((N, C, NFramesNew, H, W))
        if NFramesOri < NFramesNew:
            dataPadded[:,:,:NFramesOri,:,:] = data
        elif NFramesOri > NFramesNew:
            dataPadded = data[:,:,:NFramesNew,:,:]
        else:
            dataPadded = data
        mask = np.zeros(dataPadded.shape)
        mask[:,:,:min(NFramesOri, NFramesNew),:,:] = 1    
    elif dataType in ['TOS',  'TOSInterploated', 'TOSImage', 'TOSInterpolatedMid']:
        return dataDict
    else:
        raise ValueError(f'Unsupported padding data type: {dataType}')
    dataDict[dataType] = dataPadded
    dataDict['frameMask'] = mask
    # print('Data shape changed to '+ str(dataDict[dataType].shape))
    
    return dataDict

from scipy.interpolate import RegularGridInterpolator
def changeNSlices(dataDict:dict, NSlicesNew, dataType:str = 'strainMat', method = 'interp'):
    # Input should be merged multiple-slice data
    if method == 'padding':
        NSlicesOri = dataDict[dataType].shape[1]
        isOriSlices = np.array([False]*NSlicesNew)
        isOriSlices[:min(NSlicesOri, NSlicesNew)] = True
        # sliceLocsNew = 
        dataChanged, mask = cardiacDataSliceZeroPadding(dataDict[dataType], NSlicesNew, dataType)        
    elif method == 'interp':
        dataChanged, sliceLocsNew, isOriSlice = cardiacDataSliceInterp(dataDict[dataType], NSlicesNew, dataDict['SequenceInfo'], dataType, True)
        dataDict['sliceLocsInterped'] = sliceLocsNew
    
    dataDict[dataType] = dataChanged
    dataDict['isOriSlice'] = isOriSlice
    dataDict['NSlices'] = NSlicesNew
    # dataDict[dataDict['isOriSlice'] == False]['dataFilename'] = 'Interpreted'
    return dataDict
    

def cardiacDataSliceZeroPadding(data: np.ndarray, NSlicesNew, dataType:str = 'strainMat'):
    if NSlicesNew is None:
        return data
    if dataType in ['strainMat', 'strainMatSVD', 'strainCurve', 'TOS', 'spatialMask']:
        # Shape should be (strainMat) [1, NSlices, NSectors, NFrames]
        #                       (TOS) [1, NSlices, 1, NSectors]
        # print(data.shape)
        try:
            N, NSlicesOri, NSectors, NFrames = data.shape
        except:
            print(data.shape)
        dataPadded = np.zeros((N, NSlicesNew, NSectors, NFrames))
        if NSlicesOri < NSlicesNew:
            dataPadded[:,:NSlicesOri,:,:] = data        
        elif NSlicesOri > NSlicesNew:
            dataPadded = data[:,:NSlicesNew,:,:]
        else:
            dataPadded = data
        mask = np.zeros(dataPadded.shape)
        mask[:,:min(NSlicesOri, NSlicesNew),:,:] = 1    
    else:
        print(f'Unsupported padding type: {dataType}')
        return None, None
    return dataPadded, mask

def cardiacDataSliceInterp(data: np.ndarray, NSlicesNew: int, sliceLocs, dataType: str, keepOriLoc = False):
    # For data contain multiple slices, change slice number of padding of interpolation
    # https://www.mathworks.com/help/matlab/ref/interp3.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html#scipy.interpolate.interpn
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html#scipy.interpolate.RectBivariateSpline
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
    # [1, NSlices, NSectors, NFrames] -> [1, NewNSlices, NSectors, NFrames]    
    # Spatial Mask: [1, NSLices, H, W] -> [1, NewNSlices, H, W]
    # print(data.shape)
    # print(len(sliceLocs))
    if dataType in ['strainMat', 'strainMatSVD', 'spatialMask']:
        if (np.ndim(data) == 4 and data.shape[1] == NSlicesNew) or (np.ndim(data) == 3 and data.shape[0] == NSlicesNew):
            # If data already has enough slices, do nothing
            return data, np.array(sliceLocs), np.array([True]*NSlicesNew)
        
        if np.ndim(data) == 4 and data.shape[0] == 1:
            data = data[0]
            squeezed = True
        else:
            squeezed = False
        
        if type(sliceLocs) == list:
            sliceLocs = np.array(sliceLocs)
        
        if np.ndim(data) != 3:
            raise ValueError('Unsupported data shape: ' + str(data.shape))
        elif data.shape[0] == 1:
            raise ValueError('Data should have at least 2 slice, but get: ' + str(data.shape[0]))
        
        NSlices, NSectors, NFrames = data.shape
        xsOri, ysOri = np.arange(NSectors), np.arange(NFrames)
        # print(data.shape)
        # print(o)
        # ipdb.set_trace()

        interpFunc = RegularGridInterpolator((sliceLocs, xsOri, ysOri), data, method='linear')
                
        
        sliceLocMin, sliceLocMax = min(sliceLocs), max(sliceLocs)
        xsInterp, ysInterp = np.meshgrid(range(NFrames), range(NSectors))
        xsInterp = np.repeat(xsInterp[None,:,:], NSlicesNew, axis=0)
        ysInterp = np.repeat(ysInterp[None,:,:], NSlicesNew, axis=0)
        sliceLocsInterpRaw = np.linspace(sliceLocMin, sliceLocMax, NSlicesNew)
        
        isOriSlice = np.array([False] * NSlicesNew); isOriSlice[0] = True; isOriSlice[-1] = True
        if keepOriLoc:
            closestIndices = np.zeros(NSlices, dtype=int)
            for sliceIdx, sliceLoc in enumerate(sliceLocs):
                closestIdx = np.argmin(np.abs(sliceLocsInterpRaw - sliceLoc)) # Find closest location from original to interpolated slice
                closestIndices[sliceIdx] = closestIdx                         # 
                sliceLocsInterpRaw[closestIdx] = sliceLoc                     # Replace the interpolated slices with original one
                isOriSlice[closestIdx] = True                                 
        
        sliceLocsInterp = sliceLocsInterpRaw.reshape(NSlicesNew,1,1) * np.ones((NSlicesNew,NSectors,NFrames))
        # print(np.max(xsInterp), np.max(ysInterp), np.max(sliceLocsInterp))
        valsInterp = interpFunc(list(zip(sliceLocsInterp.flatten(),ysInterp.flatten(),xsInterp.flatten())))
        dataInterped = valsInterp.reshape((NSlicesNew, NSectors, NFrames))    
        
        if keepOriLoc:
            # Replace with original slice
            for sliceIdx, sliceLoc in enumerate(sliceLocs):
                dataInterped[closestIndices[sliceIdx]] = data[sliceIdx]
        
        if squeezed:
            dataInterped = dataInterped[None,:,:,:]
    elif dataType == 'TOS':
        # [1, NSlices, 1, NSectors] -> [1, NewNSlices, 1, NSectors]
        if (np.ndim(data) == 4 and data.shape[1] == NSlicesNew) or (np.ndim(data) == 3 and data.shape[0] == NSlicesNew):
            # If data already has enough slices, do nothing
            return data, sliceLocs, np.array([True]*NSlicesNew)
        
        if np.ndim(data) == 4 and data.shape[0] == 1:
            data = data[0,:,0,:]
            squeezed = True
        else:
            squeezed = False
        
        NSlices, NSectors = data.shape
        interpFunc = RegularGridInterpolator((sliceLocs, np.arange(NSectors)), data, method='linear')
        
        sliceLocMin, sliceLocMax = min(sliceLocs), max(sliceLocs)
        xsInterp = np.array(list(range(NSectors))*NSlicesNew)
        # sliceLocsInterp = np.array(list(np.linspace(sliceLocMin, sliceLocMax, NSlicesNew))*NSlicesNew)
        
        sliceLocsInterpRaw = np.linspace(sliceLocMin, sliceLocMax, NSlicesNew)
        isOriSlice = np.array([False] * NSlicesNew); isOriSlice[0] = True; isOriSlice[-1] = True
        if keepOriLoc:
            closestIndices = np.zeros(NSlices, dtype=int)
            for sliceIdx, sliceLoc in enumerate(sliceLocs):
                closestIdx = np.argmin(np.abs(sliceLocsInterpRaw - sliceLoc))
                closestIndices[sliceIdx] = closestIdx
                sliceLocsInterpRaw[closestIdx] = sliceLoc
                isOriSlice[closestIdx] = True
        
        sliceLocsInterp = (sliceLocsInterpRaw.reshape(NSlicesNew,1) * np.ones((NSlicesNew, NSectors))).flatten()
        
        # print(np.max(xsInterp), np.max(ysInterp), np.max(sliceLocsInterp))
        dataInterped = interpFunc(list(zip(sliceLocsInterp,xsInterp))).reshape((NSlicesNew, NSectors))
        
        if keepOriLoc:
            for sliceIdx, sliceLoc in enumerate(sliceLocs):
                dataInterped[closestIndices[sliceIdx]] = data[sliceIdx]
        
        if squeezed:
            dataInterped = dataInterped[None,:,None,:]
    else:
        raise ValueError('Unsupported Data type', dataType)
    # print(volInterped.shape)
    return dataInterped, sliceLocsInterpRaw, isOriSlice