# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:06:36 2020

@author: Jerry
"""
# import ipdb
from typing import Union
import numpy as np
import scipy.io as sio
def SVDDenoise(mat, rank):
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    s[rank:] = 0    
    return u@np.diag(s)@vh



def extractDataFromFile(dataFilename, dataTypes, labelFilename = None, labelInDataFile = True, configs=None, loadAllTypes = False):
            
    datamat = sio.loadmat(dataFilename, struct_as_record=False, squeeze_me = True)
    # maxFrame = 25
    
    try:        
        EccDatum = np.flip(datamat['TransmuralStrainInfo'].Ecc.mid.T, axis=0)[np.newaxis,np.newaxis,:,:]
    except:
        print('No TransmuralStrainInfo:', dataFilename)
        return None
    if loadAllTypes:
        # print("AA")
        dataTypes = ['strainMat', 'strainMatSVD', 'dispField', \
                     'dispFieldJacoImg', 'dispFieldJacoMat', 'dispFieldFv', 'strainImg',\
                         'SequenceInfo']
        # print(dataTypes)
    
    # print(dataTypes)
    
    data = {}
    # Load if requested
    for dataType in dataTypes:
        # print(dataType)
        if dataType == 'strainMat':
            data['strainMat'] = EccDatum
        elif dataType == 'strainMatSVD':
            data['strainMatSVD'] = np.flip(datamat['TransmuralStrainInfo'].Ecc.midSVD.T, axis=0)[np.newaxis,np.newaxis,:,:]            
        elif dataType == 'dispField':
            strainDispX = np.rollaxis(datamat['ImageInfo'].Xunwrap, -1, 0)[np.newaxis,np.newaxis,:,:]
            strainDispY = np.rollaxis(datamat['ImageInfo'].Yunwrap, -1, 0)[np.newaxis,np.newaxis,:,:]        
            data['dispField'] = np.concatenate((strainDispX, strainDispY), axis=-4)
            data['dispField'][np.isnan(data['dispField'])] = 0
        elif dataType == 'dispFieldMat':
            dispX = datamat['DisplacementInfo'].dX[None,None,:,:]
            dispY = datamat['DisplacementInfo'].dY[None,None,:,:]
            data['dispField'] = np.concatenate([dispX, dispY], axis=1)
        elif dataType == 'dispFieldJacoImg':
            data['dispFieldJacoImg'] = np.rollaxis(datamat['ImageInfo'].dispFieldJaco, -1, 0)[np.newaxis,np.newaxis,:,:]
        elif dataType == 'dispFieldJacoMat':
            data['dispFieldJacoMat'] = np.flip(datamat['ImageInfo'].dispFieldJacoSeged, axis=0)[np.newaxis,np.newaxis,:,:]            
        elif dataType == 'strainImg':
            data['strainImg'] = np.rollaxis(datamat['StrainInfo'].CCImgs, -1, 0)[np.newaxis,np.newaxis,:,:]        
        elif dataType == 'dispFieldFv':
            # [FvPerimeter*Nlayer,NFrame] -> [FvPerimeter, Nlayer,NFrame]             
            FvPerimeter = datamat['AnalysisInfo'].fv.FvPerimeter
            Nlayer = datamat['AnalysisInfo'].fv.Nlayer
            dispX = np.reshape(datamat['DisplacementInfo'].fvX, (FvPerimeter,Nlayer,-1), order = 'F')
            dispY = np.reshape(datamat['DisplacementInfo'].fvY, (FvPerimeter,Nlayer,-1), order = 'F')
            # -> [NFrame,FvPerimeter, Nlayer] -> [1,1,NFrame,FvPerimeter, Nlayer]
            dispX = np.rollaxis(dispX, -1,0)[None,None,:,:,:]
            dispY = np.rollaxis(dispY, -1,0)[None,None,:,:,:]
            data['dispFieldFv'] = np.concatenate((dispX,dispY), axis=1)            
        elif dataType == 'SequenceInfo':
            data['SequenceInfo'] = datamat['SequenceInfo'][0,0].SliceLocation        
        elif dataType in ['TOS', 'TOSImage', 'StrainInfo', 'AnalysisFv']:
            pass
        else:
            raise ValueError(f'Unrecognized data type: {dataType}')
    
    # Load if exist
    if labelFilename != None and labelInDataFile == False:
        if dataType in ['TOSInterploated', 'TOSImage']:
            raise ValueError(f"{dataType} isn't included in raw TOS file")
        data['TOS'] = sio.loadmat(labelFilename)['xs']
    elif labelFilename == None and labelInDataFile == True:
        if 'TOSAnalysis' in datamat.keys():
            data['TOS'] = datamat['TOSAnalysis'].TOS[np.newaxis, :]
            data['TOSInterploated'] = datamat['TOSAnalysis'].TOSInterploated[np.newaxis, :]
            data['TOSImage'] = datamat['TOSAnalysis'].TOSImage[np.newaxis, np.newaxis, :,:]
        else:
            print('No TOS data in ' + dataFilename)
    elif labelFilename == None and labelInDataFile == False:
        pass
    else:
        raise ValueError('Should not provide both label filename and labelInDataFile = True')
    
    if 'StrainInfo' in datamat.keys():        
        # print('Have StrainInfo')
        data['spatialMask'] = datamat['StrainInfo'].Mask[np.newaxis,np.newaxis,:,:]
    if 'AnalysisInfo' in datamat.keys():        
        # print('Have Analysis FV')
        data['AnalysisFv'] = datamat['AnalysisInfo'].fv
        data['hasScar'] = datamat['AnalysisInfo'].hasScar
    
    data['dataFilename'] = dataFilename
    data['labelFilename'] = labelFilename
    
    for key in data.keys():
        if type(data[key]) == np.ndarray:
            data[key] = data[key].astype(np.float32)
            
    return data


def extractDataGivenFilenames(dataFilenames, labelFilenames = None, dataTypes = {'strainMat', 'TOS'}, labelInDataFile = True, configs = None, loadAllTypes = False):
    if labelFilenames is not None and len(dataFilenames) != len(labelFilenames):
        raise ValueError(f'Length of data filenames {len(dataFilenames)} and label {len(labelFilenames)} should be the same')
    dataFilenamesValid, labelFilenamesValid = [], []
    data = []
    for fileIdx, dataFilename in enumerate(dataFilenames):
        labelFilename = None if labelFilenames == None else labelFilenames[fileIdx]        
        datum = extractDataFromFile(dataFilename, dataTypes, labelFilename, labelInDataFile, configs, loadAllTypes)
        
        if datum is None:
            continue
        else:
            data.append(datum)        
            dataFilenamesValid.append(dataFilename)
            labelFilenamesValid.append(labelFilename)
                            
    return data, dataFilenamesValid, labelFilenamesValid

def cardiacDataFramePadding(data, dataType, configs):
    paddingVal = configs.get('paddingVal', 0)
    paddingMethod = configs.get('paddingMethod', 'zero')
    frameCount = configs.get('frameCount', None)    
    if frameCount is None:
        return data
    if dataType in ['strainMat', 'strainMatSVD', 'strainCurve','dispFieldJacoMat']:
        # Shape should be [1, 1, NSectors, NFrames]
        # print(data.shape)
        try:
            N, C, NSectors, NFrames = data.shape
        except:
            print(data.shape)
        dataPadded = np.zeros((N, C, NSectors, frameCount))
        if NFrames < frameCount:
            if paddingMethod == 'zero':                
                # dataPadded = np.concatenate((data, paddingVal*np.ones((N, C, NSectors, frameCount - NFrames))), axis=-1)
                dataPadded[:,:,:,:NFrames] = data
            elif paddingMethod == 'circular':
                n_full_fill = frameCount // NFrames
                fflip = False                         # Flip in frame dimension
                for fillIdx in range(n_full_fill):
                    fillStartIdx = fillIdx*NFrames
                    fillEndIdx = (fillIdx+1)*NFrames
                    if fflip:
                        dataPadded[:,:,:,fillStartIdx:fillEndIdx] = np.flip(data, axis=-1)
                    else:
                        dataPadded[:,:,:,fillStartIdx:fillEndIdx] = data
                    fflip = not fflip
                if frameCount % NFrames != 0:
                    if fflip:
                        dataPadded[:,:,:,fillEndIdx:] = np.flip(data, axis=-1)[:,:,:,:frameCount-fillEndIdx]
                    else:
                        dataPadded[:,:,:,fillEndIdx:] = data[:,:,:,:frameCount-fillEndIdx]
        elif NFrames > frameCount:
            dataPadded = data[:,:,:,:frameCount]
        else:
            dataPadded = data
        mask = np.zeros(dataPadded.shape)
        mask[:,:,:,:min(NFrames, frameCount)] = 1
    elif dataType in ['dispField', 'dispFieldFv']:
        # Shape should be [N, 2, NFrames, H, W]
        N, C, NFrames, H, W = data.shape
        dataPadded = np.zeros((N, C, frameCount, H, W))
        if NFrames < frameCount:
            if paddingMethod == 'zero':
                dataPadded[:,:,:NFrames,:,:] = data
            elif paddingMethod == 'circular':
                n_full_fill = frameCount // NFrames
                fflip = False
                for fillIdx in range(n_full_fill):
                    fillStartIdx = fillIdx*NFrames
                    fillEndIdx = (fillIdx+1)*NFrames
                    if fflip:
                        dataPadded[:,:,fillStartIdx:fillEndIdx,:,:] = np.flip(data, axis = 2)
                    else:
                        dataPadded[:,:,fillStartIdx:fillEndIdx,:,:] = data
                    fflip = not fflip
                if frameCount % NFrames != 0:
                    if fflip:
                        dataPadded[:,:,fillEndIdx:,:,:] = np.flip(data, axis = 2)[:,:,:frameCount-fillEndIdx,:,:]
                    else:
                        dataPadded[:,:,fillEndIdx:,:,:] = data[:,:,:frameCount-fillEndIdx,:,:]
            
        elif NFrames > frameCount:
            dataPadded = data[:,:,:frameCount,:,:]
        else:
            dataPadded = data
        mask = np.zeros(dataPadded.shape)
        mask[:,:,:min(NFrames, frameCount),:,:] = 1    
    elif dataType in ['TOS',  'TOSInterploated', 'TOSImage']:
        return data, np.ones(data.shape)
    else:
        raise ValueError(f'Unsupported padding data type: {dataType}')
    return dataPadded, mask

from scipy.interpolate import RegularGridInterpolator
def cardiacDataSliceInterp(data: np.ndarray, NSlicesNew: int, sliceLocs, dataType: str, keepOriLoc = False):
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
        try:
            interpFunc = RegularGridInterpolator((sliceLocs, xsOri, ysOri), data, method='linear')
        except:
            ipdb.set_trace()
            nouse = 0
                
        
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


def cardiacDataSlicePadding(data, dataType, configs):
    paddingVal = configs.get('paddingVal', 0)
    paddingMethod = configs.get('paddingMethod', 'zero')
    sliceCount = configs.get('sliceCount', np.max([datum.shape[1] for datum in data]))
    if sliceCount is None:
        return data
    if dataType in ['strainMat', 'strainMatSVD', 'strainCurve', 'TOS', 'spatialMask']:
        # Shape should be [1, NSlices, NSectors, NFrames]
        # print(data.shape)
        try:
            N, NSlices, NSectors, NFrames = data.shape
        except:
            print(data.shape)
        dataPadded = np.zeros((N, sliceCount, NSectors, NFrames))
        if NSlices < sliceCount:
            if paddingMethod == 'zero':                
                # dataPadded = np.concatenate((data, paddingVal*np.ones((N, C, NSectors, frameCount - NFrames))), axis=-1)
                dataPadded[:,:NSlices,:,:] = data
            elif paddingMethod == 'circular':
                n_full_fill = sliceCount // NSlices
                fflip = False                         # Flip in frame dimension
                for fillIdx in range(n_full_fill):
                    fillStartIdx = fillIdx*NSlices
                    fillEndIdx = (fillIdx+1)*NSlices
                    if fflip:
                        dataPadded[:,fillStartIdx:fillEndIdx,:,:] = np.flip(data, axis=1)
                    else:
                        dataPadded[:,fillStartIdx:fillEndIdx,:,:] = data
                    fflip = not fflip
                if sliceCount % NSlices != 0:
                    if fflip:
                        dataPadded[:,fillEndIdx:,:,:] = np.flip(data, axis=1)[:,:sliceCount-fillEndIdx,:,:]
                    else:
                        dataPadded[:,fillEndIdx:,:,:] = data[:,:sliceCount-fillEndIdx,:,:]
        elif NSlices > sliceCount:
            dataPadded = data[:,:sliceCount,:,:]
        else:
            dataPadded = data
        mask = np.zeros(dataPadded.shape)
        mask[:,:min(NSlices, sliceCount),:,:] = 1
    elif dataType == 'TOS':
        # (OLD) [1,1,NSlices,NFrames] -> [1,1,sliceCount, NFrames]
        # [1, NSlices, ]
        pass
        # try:
        #     _, _, NSlices, NFrames = data.shape
        # except:
        #     print(data.shape)
        # dataPadded = np.zeros((1, 1, sliceCount, NFrames))
        # if NSlices < sliceCount:
        #     if paddingMethod == 'zero':                
        #         # dataPadded = np.concatenate((data, paddingVal*np.ones((N, C, NSectors, frameCount - NFrames))), axis=-1)
        #         dataPadded[:,:NSlices,:,:] = data
        #     elif paddingMethod == 'circular':
        #         raise ValueError('Unsupported!')
        # elif NSlices > sliceCount:
        #     dataPadded = data[:,:sliceCount,:,:]
        # else:
        #     dataPadded = data
        # mask = np.zeros(dataPadded.shape)
        # mask[:,:min(NSlices, sliceCount),:,:] = 1
    else:
        print(f'Unsupported padding type: {dataType}')
        return None, None
    return dataPadded, mask
      
# def trainTestSplit0(NSlices, trainRatio, splitByPatient = False, NPatients = None, patientIDs = None):
#     if splitByPatient:
#         N = NPatients
#     else:
#         N = NSlices
#     indices_random = np.arange(N)
#     np.random.shuffle(indices_random)
#     train_indices_random = indices_random[:int(N*trainRatio)]
#     test_indices_random = indices_random[int(N*trainRatio):]
     
#     patientValidNames = [dataFilename.split('Jerry/')[1].split('/mat')[0] for dataFilename in dataFilenamesValid]
#     patientValidNamesUnique = list(dict.fromkeys(patientValidNames))
#     NPatients = len(patientValidNamesUnique)
#     trainPatientNamesFixed = patientValidNamesUnique[:int(trainRatio*NPatients)]
#     testPatientNamesFixed = patientValidNamesUnique[int(trainRatio*NPatients):]
#     train_indices_fixedPatient = np.arange(N)[np.array([name in trainPatientNamesFixed for name in patientValidNames])]
#     test_indices_fixedPatient = np.arange(N)[np.array([name in testPatientNamesFixed for name in patientValidNames])]
    
#     patientValidNamesUniqueRandom = patientValidNamesUnique.copy()
#     np.random.shuffle(patientValidNamesUniqueRandom)
#     trainPatientNamesRandom = patientValidNamesUniqueRandom[:int(trainRatio*NPatients)]
#     testPatientNamesRandom = patientValidNamesUniqueRandom[int(trainRatio*NPatients):]
#     train_indices_randomPatient = np.arange(N)[np.array([name in trainPatientNamesRandom for name in patientValidNames])]
#     test_indices_randomPatient = np.arange(N)[np.array([name in testPatientNamesRandom for name in patientValidNames])]
#     return train_indices_random, test_indices_random, \
#         train_indices_fixedPatient, test_indices_fixedPatient,\
#         train_indices_randomPatient, test_indices_randomPatient

# def trainTestSplit(N, trainRatio = 0.8, random:bool = True, splitByID:bool = False, dataIDs:list = None):
# def trainTestSplit(N, trainRatio = 0.8, method:str = 'randomSlice', dataIDs:list = None):
def trainTestSplit(N, trainRatio = 0.8, orgBy = 'slice', splBy = 'slice', random = 'random', 
                   dataIDs:list = None, mixRatio = 0.5, ifDataHaveTOS: Union[np.ndarray, None] = None,
                   forceTestNames = []):
    # orgBy: how will the slices later will be organized. 
    #        If orgBy=='patient', N should be number of patients instead of number of slices
    # splBy: how to split the slices
    # random: random, fixed, mixed (part random part fixed), allLabeled(all labed data as training and all unlabeled as test)
    NTr = int(N*trainRatio)
    NTe = N - NTr
    indicesN = np.arange(N)
    indices300 = [161, 261, 136, 132, 172, 279, 1, 236, 162, 210, 253, 52, 86, 193, 165, 185, 297, 249, 200, 196, 180, 284, 245, 235, 262, 240, 272, 266, 265, 14, 176, 11, 8, 182, 82, 223, 57, 201, 96, 10, 92, 43, 224, 233, 45, 58, 194, 267, 248, 42, 17, 179, 87, 120, 250, 146, 192, 20, 110, 89, 292, 299, 283, 152, 37, 264, 107, 154, 275, 295, 216, 149, 195, 55, 163, 198, 277, 212, 206, 83, 263, 214, 72, 7, 71, 268, 44, 191, 70, 31, 221, 252, 153, 202, 213, 290, 102, 209, 144, 130, 150, 48, 2, 139, 23, 88, 25, 119, 12, 121, 125, 123, 168, 173, 169, 134, 9, 53, 94, 111, 81, 222, 228, 157, 244, 255, 260, 35, 205, 65, 259, 6, 286, 4, 241, 24, 108, 276, 282, 16, 273, 293, 158, 247, 294, 204, 142, 137, 109, 75, 126, 67, 133, 188, 106, 104, 33, 296, 135, 298, 187, 100, 101, 131, 19, 208, 122, 26, 291, 78, 3, 127, 114, 289, 164, 156, 63, 189, 147, 174, 231, 242, 50, 79, 18, 155, 117, 0, 258, 178, 129, 99, 32, 227, 90, 197, 238, 219, 39, 181, 51, 64, 167, 175, 186, 203, 61, 278, 66, 116, 30, 270, 271, 257, 97, 115, 98, 141, 113, 124, 183, 287, 128, 74, 85, 184, 22, 91, 254, 38, 199, 226, 170, 54, 68, 46, 232, 29, 274, 288, 73, 243, 160, 190, 256, 15, 251, 95, 171, 105, 36, 229, 246, 28, 49, 218, 80, 234, 177, 281, 93, 269, 211, 230, 285, 166, 13, 112, 217, 62, 159, 220, 60, 145, 215, 151, 27, 148, 225, 77, 41, 143, 103, 59, 84, 237, 239, 138, 118, 69, 207, 140, 40, 280, 47, 34, 21, 76, 56, 5]
    if (orgBy == 'slice' and splBy == 'slice') or orgBy == 'patient':
        if random == 'allLabeled':
            # print(ifDataHaveTOS)
            train_indices = np.where(ifDataHaveTOS)[0]
            test_indices = np.where(~ifDataHaveTOS)[0]
            # print(train_indices)
        else:
            if random == 'random':
                indices = [idx for idx in indices300 if idx < N]
            elif random == 'fixed':
                indices = indicesN
            elif random == 'mixed':
                indices = np.append(\
                                    np.array([idx for idx in indices300 if idx < NTr+int(NTe*mixRatio)]),\
                                    indicesN[NTr+int(NTe*mixRatio):])        
            train_indices = indices[:NTr]
            test_indices  = indices[NTr:]
        
        
        
    elif splBy == 'patient' and random in ['random', 'fixed']:
        if len(dataIDs) != N:
            raise ValueError(f'Length of dataIDs should be N={N} instead of {len(dataIDs)}')
        uniqueIDs = list(dict.fromkeys(dataIDs))

        if random == 'random':
            np.random.shuffle(uniqueIDs)
        
        NID = len(uniqueIDs)
        uniqueIDsTr = uniqueIDs[:int(NID*trainRatio)]
        uniqueIDsTe = uniqueIDs[int(NID*trainRatio):]
        train_indices = indicesN[np.array([ID in uniqueIDsTr for ID in dataIDs])]
        test_indices = indicesN[np.array([ID in uniqueIDsTe for ID in dataIDs])]
    else:
        raise ValueError('Unsupported', orgBy, splBy, random)
        
    return train_indices, test_indices
    

def mergeDataByPatient(data:list, patientIDs:list = None, reorderByLoc = False):
    # print(reorderByLoc)
    if patientIDs is None:
        patientIDs = [datum['dataFilename'].split('Jerry/')[1].split('/mat')[0] for datum in data]
    #NPatients = len(list(set(patientIDs)))
    dataConcated = []
    for patientID in list(dict.fromkeys(patientIDs)):
        dataOfPatient = [data[idx] for idx in range(len(data)) if patientIDs[idx] == patientID]

        dataOfPatientMerged = mergeData(dataOfPatient, list(dataOfPatient[0].keys()), reorderByLoc = reorderByLoc)
        
        if dataOfPatientMerged != {}:
            dataOfPatientMerged['patientID'] = patientID
            dataConcated.append(dataOfPatientMerged)
        else:
            print('failed to merge')
    return dataConcated

def mergeData(data:list, dataTypes:list, padding = True, reorderByLoc = False):
    # Merge several data into a new single data
    # print(reorderByLoc)
    if reorderByLoc:
        spatialLocs = [datum['SequenceInfo'] for datum in data]
        # print(np.argsort(spatialLocs))
        data = [data[idx] for idx in np.argsort(spatialLocs)]
    dataMerged = {}
    for dataType in dataTypes:
        if dataType in ['strainMat','strainMatSVD','dispFieldJacoMat', 'spatialMask', 'TOSImage']:
            # [1, 1, NS, NF] -> [1, NSlice, NSector, NFrame]
            try:
                dataMerged[dataType] = np.concatenate([datum[dataType] for datum in data], axis=1)
            except:
                # try:
                # print('Shapes:', [datum[dataType].shape for datum in data])
                frameCountMax = np.max([datum[dataType].shape[-1] for datum in data])
                paddingConfig = {'frameCount':frameCountMax,'paddingMethod':'zero'}
                dataMerged[dataType] = np.concatenate([cardiacDataFramePadding(datum[dataType], dataType, paddingConfig)  for datum in data][0], axis=1)
                # print('Merged Shapes:', dataMerged[dataType].shape)
                # except:                    
                    # print('failed:',dataType)
                    # print([datum[dataType].shape for datum in data])
                    # break
                    # continue
                
                
        elif dataType in ['TOS', 'TOSInterploated']:
            # (OLD) [1,NSector] -> [1, 1, NSlice, NSector]
            # [1,NSector] -> [1,NSlices, 1, NSector]
            #dataMerged[dataType] = np.concatenate([datum[dataType][:,None,None,:] for datum in data], axis=2)
            dataMerged[dataType] = np.concatenate([datum[dataType][:,None,None,:] for datum in data], axis=1)
        elif dataType in ['dataFilename', 'labelFilename', 'SequenceInfo']:
            dataMerged[dataType] = [datum[dataType] for datum in data]
        else:
            pass
            # print(f'Unsupported type to merge: {dataType}')
    return dataMerged

def getConcatedData(data:list, dataType, paddingConfig):
    # Concate several data into single variable
    if len(data) == 0:
        return None, None, None
    others = {}
    if paddingConfig.get('slice', False):
        if paddingConfig['slice'] == 'padding':# or dataType not in ['strainMat', 'strainMatSVD']:
            sliceMax = np.max([datum[dataType].shape[1] for datum in data])
            paddingConfig['sliceCount'] = sliceMax
            data, sliceMasksRaw = zip(*[cardiacDataSlicePadding(datum[dataType], dataType, paddingConfig) for datum in data])
        elif paddingConfig['slice'] in ['interp', 'interpolate']:
            sliceMax = paddingConfig.get('sliceCount', np.max([datum[dataType].shape[1] for datum in data]))
            # data = [cardiacDataSliceInterp(datum[dataType], sliceMax, datum['SequenceInfo'], dataType) for datum in data]
            # print(len([cardiacDataSliceInterp(datum[dataType], sliceMax, datum['SequenceInfo'], dataType) for datum in data][0]))
            data, sliceLocsInterped, isOriSlice = zip(*[cardiacDataSliceInterp(datum[dataType], sliceMax, datum['SequenceInfo'], dataType, True) for datum in data])
            sliceMasksRaw = [np.ones(datum.shape) for datum in data]
            others['sliceLocsInterped'] = sliceLocsInterped
            others['isOriSlice'] = isOriSlice
    else:
        data = [datum[dataType] for datum in data]
        sliceMasksRaw = [np.ones(datum.shape) for datum in data]
    
    if paddingConfig.get('frame', True):
        #data, frameMasks = [cardiacDataFramePadding(datum, dataType, paddingConfig)[0] for datum in data]
        data, frameMasks = zip(*[cardiacDataFramePadding(datum, dataType, paddingConfig) for datum in data])
        sliceMasksFPadded = [cardiacDataFramePadding(mask, dataType, paddingConfig)[0] for mask in sliceMasksRaw]
    else:
        frameMasks = [np.ones(datum.shape) for datum in data]
        sliceMasksFPadded = sliceMasksRaw
    
    dataConcat, maskConcat = np.concatenate(data, axis=0), np.concatenate(sliceMasksFPadded, axis=0)
    if paddingConfig.get('slice', False) and np.ndim(dataConcat)==4:
        dataConcat = dataConcat[:,None,:,:,:]
        maskConcat = maskConcat[:,None,:,:,:]
    return dataConcat, maskConcat, others#*np.concatenate(frameMasks, axis=0)

# def concatCardiacData(data: list, dataTypes: list, padding = True, configs = {'frameCount': None, 'paddingVal': 0}):
#     dataConcated = {}
#     for dataIdx in range(len(data)):
#         for dataType in dataTypes:
#             datum = cardiacDataFramePadding(data[dataIdx][dataType], dataType, configs)
#             if dataIdx == 0:
#                 dataConcated[dataType] = datum
#             else:
#                 dataConcated[dataType] = np.concatenate((dataConcated[dataType], datum), axis=0)
#     return dataConcated
            
def getFilenamesGivenPath(bulk_path = '../../Dataset/CRT_TOS_Data_Jerry/', loadProcessed = True):
    import os    
    data_filenames = []
    TOS_filenames  = []
    patient_IDs    = []
    matFolder = '/mat_processed/' if loadProcessed else '/mat/'
    list_dataset_with_paths = [f.path for f in os.scandir(bulk_path) if f.is_dir() and 'SET' in f.name]
    patient_ID = 0
    for dataset_path in list_dataset_with_paths:
        list_patients = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
        for patient_path in list_patients:
            try:                
                list_slices_data_unsorted = [f.path for f in os.scandir(patient_path + matFolder) if f.is_file()]
                #list_slices_data = [f.path for f in os.scandir(patient_path + '/mat/') if f.is_file()]
                #list_slices_TOS_unsorted  = [f.path for f in os.scandir(patient_path + '/TOS/') if f.is_file()]# and 'processed' in f.name
                if os.path.exists(patient_path + '/TOS/'):
                    list_slices_TOS_unsorted = [f.path for f in os.scandir(patient_path + '/TOS/') if f.is_file()]
                else:
                    list_slices_TOS_unsorted = []
                list_slices_data_order = np.argsort([fname.lower() for fname in list_slices_data_unsorted])
                list_slices_TOS_order = np.argsort([fname.lower() for fname in list_slices_TOS_unsorted])
                list_slices_data = [list_slices_data_unsorted[fidx] for fidx in list_slices_data_order]
                list_slices_TOS = [list_slices_TOS_unsorted[fidx] for fidx in list_slices_TOS_order]                
            except:
                print(f'failed load {patient_path}')
                continue
            if len(list_slices_TOS)!=0 and (len(list_slices_data) != len(list_slices_TOS)):
                print(f'# of files mismatch at {patient_path}')
                continue
            data_filenames += list_slices_data
            TOS_filenames  += list_slices_TOS if len(list_slices_TOS)!=0 else [None]*len(list_slices_data)
            patient_IDs    += [patient_ID] * len(list_slices_data)
            patient_ID     += 1
            
    return data_filenames, TOS_filenames, patient_IDs

if __name__ == '__main__':
    
    dataConfig = {
        #'inputType': 'dispFieldJacoImg',
        #'inputType': 'strainMatSVD',
        # 'inputType': 'myocardiacImgs',
        'inputType': 'strainMat',
        # 'outputType': 'TOS',
        'outputType': 'TOSImage',
        # 'augmentation': {'shiftY': 10}
        'augmentation': {}
        # 'augmentation': {'shiftY':10, 'shiftX':10}
        }
    
    dataFilenames, labelFilenames = getFilenamesGivenPath(loadProcessed = True)
    dmat = sio.loadmat(dataFilenames[12], struct_as_record=False, squeeze_me = True)
    TOSmat = sio.loadmat(labelFilenames[12], struct_as_record=False, squeeze_me = True)    
    
    data, dataFilenamesValid, labelFilenamesValid = extractDataGivenFilenames(dataFilenames[:20], labelFilenames = None, dataTypes = {'strainMat', 'TOS'}, labelInDataFile = True, configs = None)
    frameCountMax = np.max([datum['strainMat'].shape[-1] for datum in data])
    paddingConfig = {'frameCount':frameCountMax, 'paddingMethod': 'circular'}
    EccMat = np.concatenate([cardiacDataFramePadding(datum['strainMat'], 'strainMat', paddingConfig)[0] for datum in data], axis=0)
    EccMaskMat = np.concatenate([cardiacDataFramePadding(datum['strainMat'], 'strainMat', paddingConfig)[1] for datum in data], axis=0)
    # EccMat = concatCardiacData(data, ['strainMat'], {'frameCount':frameCountMax})['strainMat']
    TOSData = np.concatenate([datum['TOS'] for datum in data], axis=0)
    
    # TOSImageData = np.concatenate([datum['TOSImage'] for datum in data], axis=0)
    
    
    
    # Visualize to check
    import matplotlib.pyplot as plt
    def visTOS(TOS, strainMat, vmin = -0.2, vmax = 0.2, title = ''):
        TOSGrid = np.flip((TOS / 17).flatten() - 0.5)
        fig, axe = plt.subplots()
        c = axe.pcolor(strainMat, cmap='jet', vmin = vmin, vmax = vmax)
        axe.plot(TOSGrid,np.arange(len(TOS))+0.5)
        axe.set_title(title)
        fig.colorbar(c, ax=axe)
    
    #for cIdx in [0,10,30,50,70,90]:
    # for cIdx in [5,15,35,55,75,95]:
    for cIdx in [0,5,10,15,19]:
        #visTOS(np.squeeze(TOSData[cIdx,:]),np.squeeze(EccData[cIdx,:]), title = f'Raw Midwall Ecc,\n slice location = {sliceLocations[cIdx]}')        
        if dataConfig['inputType'] in ['strainMat', 'strainMatSVD', 'dispFieldJaco']:
            visTOS(np.squeeze(TOSData[cIdx,:]),np.squeeze(EccMat[cIdx,:]), \
                   #vmin = 0, vmax = 2,\
                   title = f'Denoised Midwall Ecc')
        # visTOS(np.squeeze(TOSData[cIdx,:]),SVDDenoise(np.squeeze(EccData[cIdx,:]),3))
        
    def visTOsS(TOSs, strainMat = None, legends = None, title = None):
        fig, axe = plt.subplots()
        if strainMat is not None:
            axe.pcolor(strainMat, cmap='jet', vmin = -0.2, vmax = 0.2)
        for idx, TOS in enumerate(TOSs):
            TOSGrid = np.flip((TOS / 17).flatten() - 0.5)
            line, = axe.plot(TOSGrid,np.arange(len(TOS))+0.5)
            if legends is not None:
                line.set_label(legends[idx])
            if title is not None:
                axe.set_title(title)
        axe.legend()    

    if 'TOSImage' in data[0].keys():
        from utils.cardiacUtils import mat2recvfv, rectfv2TOS
        TOSFvVals = [mat2recvfv(datum['AnalysisFv'].faces, datum['AnalysisFv'].vertices, np.squeeze(datum['TOSImage'])) for datum in data]
        TOSRestored = [rectfv2TOS(TOSFvVals[dataIdx], data[dataIdx]['AnalysisFv'].sectorid, data[dataIdx]['AnalysisFv'].layerid) for dataIdx in range(len(TOSFvVals))]
        for cIdx in [0,5,10,15,19]:
            visTOsS([data[cIdx]['TOS'].flatten(), TOSRestored[cIdx].flatten()], np.squeeze(data[cIdx]['strainMat']), ['GT','Restored'])