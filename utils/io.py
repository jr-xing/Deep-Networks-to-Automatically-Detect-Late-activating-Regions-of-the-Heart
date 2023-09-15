# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:11:33 2019

@author: remus
"""
import numpy as np
import matplotlib.pyplot as plt
def saveStrainDataFig2File(strainData: np.ndarray, TOSData: list, save_filename, \
                           legends = None, title = None, subtitles = None, \
                           vmin = -0.2, vmax = 0.2, inverseTOS = True, markPeak = True, truncate = None):
    # strainData: [NSector, NFrame] or [NSlice, NSector, NFrame]
    if len(np.squeeze(strainData.shape)) == 2:
        saveStrainMatFig2File(strainData, TOSData, save_filename,\
                              legends, title, subtitles, vmin, vmax, inverseTOS, markPeak, truncate)
    elif len(np.squeeze(strainData.shape)) == 3:
        # Throw away padding slices
        if subtitles is not None:
            strainData = strainData[:len(subtitles)]
        saveStrainSlicesFig2File(strainData, TOSData, save_filename,\
                              legends, title, subtitles, vmin, vmax, inverseTOS, markPeak, truncate)                
    else:
        pass
        
    

def saveStrainMatFig2File(strainMat, TOSs, save_filename, legends = None, title = None, subtitle = None, vmin = -0.2, vmax = 0.2, 
                          inverseTOS = True, markPeak = True, truncate = None):
    plt.ioff()
    fig, axe = plt.subplots()
    truncate = strainMat.shape[1] if truncate is None else truncate
    if strainMat is not None:
        axe.pcolor(strainMat[:,:truncate], cmap='jet', vmin = vmin, vmax = vmax)
    # colors = ['#000000','#0485d1','#ff9408']
    colors = ['#000000','#0485d1','#FF4720']
    for idx, TOS in enumerate(TOSs):                
        if TOS is None:
            continue
        TOSGrid = np.flip((TOS / 17).flatten() - 0.5)
        # peakTOSVal = np.max(TOSGrid)
        # peakSectorIds = np.arange(1,19)[peakTOSVal - TOSGrid < 1e-3]
        # print(peakSectorIds)
        
        
        line, = axe.plot(TOSGrid,np.arange(len(TOS))+0.5,color = colors[idx], linewidth=4)
        if markPeak:
            peakSectorIdx = np.argmax(TOSGrid)
            plt.plot(TOSGrid[peakSectorIdx],peakSectorIdx+0.5, '_', color = colors[idx])
        if legends is not None:
            line.set_label(legends[idx])
        if title is not None:
            axe.set_title(subtitle)
        axe.set_yticks(np.arange(1,19)-0.5)
        axe.set_yticklabels(np.arange(1,19))
    axe.legend(prop={'size': 15})
    if title is not None: fig.suptitle(title, y=0.02)
    fig.savefig(save_filename, bbox_inches='tight')   # save the figure to file
    plt.close(fig)        

import matplotlib as mpl
def saveStrainSlicesFig2File(strainVol, TOSVols, save_filename, legends = None, \
                             title = None, subtitles = None,\
                             vmin = -0.2, vmax = 0.2, inverseTOS = True,\
                             markPeak = True, truncate = None):
    # Save slices of one patient to subplots
    # strainVol: [NSlice, NSector, NFrame] = e.g. [9, 18, 61]
    # TOSVOL: [NSlice, NSector] = e.g. [9, 18]
    # print(TOSVols[0])
    mpl.rcParams.update(mpl.rcParamsDefault)
    NSlices = strainVol.shape[0]
    # NRows, NCols = 2, 5
    NCols = 5
    NRows = 1 if NSlices <= NCols else 2
    figHeight = 8 if NRows == 2 else 4
    figWidth = 20    
    plt.ioff()
    # plt.axis('off')
    #fig, axs = plt.subplots(NRows, NCols, figsize=(10*NRows,1.5*NCols))
    fig, axs = plt.subplots(NRows, NCols, figsize=(figWidth,figHeight))
    plt.subplots_adjust(hspace=0.5)
    axs = axs.ravel()
    colors = ['#0485d1','#ff9408']
    titleColors = ['#000000', '#3b8bd1']
    for sliceIdx in range(NSlices):
        axe = axs[sliceIdx]        
        strainMat = np.squeeze(strainVol[sliceIdx,:])
        axe.pcolor(strainMat, cmap='jet', vmin = vmin, vmax = vmax)
        for TOSIdx, TOSVol in enumerate(TOSVols):
            TOS = np.squeeze(TOSVol[sliceIdx,:])
            TOSGrid = np.flip((TOS / 17).flatten() - 0.5) if inverseTOS else (TOS / 17).flatten() - 0.5            
            line, = axe.plot(TOSGrid,np.arange(len(TOS))+0.5,color = colors[TOSIdx])
            if markPeak:
                peakSectorIdx = np.argmax(TOSGrid)
                axe.plot(TOSGrid[peakSectorIdx],peakSectorIdx+0.5, '_', color = colors[TOSIdx])
            if legends is not None:
                line.set_label(legends[TOSIdx])
            if subtitles is not None:
                if 'train' in subtitles[sliceIdx]:
                    subtitleColor = titleColors[0]
                else:
                    subtitleColor = titleColors[1]
                # print(subtitleColor)
                axe.set_title(subtitles[sliceIdx], color = subtitleColor)
            axe.set_yticks(np.arange(1,19)-0.5)
            axe.set_yticklabels(np.arange(1,19))
        axe.legend()
    for sliceIdx in range(NSlices, NRows*NCols):
        axe = axs[sliceIdx]
        axe.axis("off")
    if title is not None: fig.suptitle(title, y=0.02)
    fig.savefig(save_filename, bbox_inches='tight')   # save the figure to file
    plt.close(fig)

def saveStrainSlicesFig2FileOLD(strainVol, TOSVols, save_filename, legends = None, \
                             title = None, subtitles = None,\
                             vmin = -0.2, vmax = 0.2):
    # Save slices of one patient to subplots
    # strainVol: [NSlice, NSector, NFrame] = e.g. [9, 18, 61]
    # TOSVOL: [NSlice, NSector] = e.g. [9, 18]
    NSlices = strainVol.shape[0]
    # NRows, NCols = 2, 5
    NCols = 5
    NRows = 1 if NSlices <= NCols else 2
    figHeight = 8 if NRows == 2 else 4
    figWidth = 20    
    plt.ioff()
    # plt.axis('off')
    #fig, axs = plt.subplots(NRows, NCols, figsize=(10*NRows,1.5*NCols))
    fig, axs = plt.subplots(NRows, NCols, figsize=(figWidth,figHeight))
    plt.subplots_adjust(hspace=0.5)
    axs = axs.ravel()
    colors = ['#0485d1','#ff9408']
    titleColors = ['#000000', '#3b8bd1']
    for sliceIdx in range(NSlices):
        axe = axs[sliceIdx]        
        strainMat = np.squeeze(strainVol[sliceIdx,:])
        axe.pcolor(strainMat, cmap='jet', vmin = vmin, vmax = vmax)
        for TOSIdx, TOSVol in enumerate(TOSVols):
            TOS = np.squeeze(TOSVol[sliceIdx,:])
            TOSGrid = np.flip((TOS / 17).flatten() - 0.5)
            peakSectorIdx = np.argmax(TOSGrid)
            line, = axe.plot(TOSGrid,np.arange(len(TOS))+0.5,color = colors[TOSIdx])
            axe.plot(TOSGrid[peakSectorIdx],peakSectorIdx+0.5, '_', color = colors[TOSIdx])
            if legends is not None:
                line.set_label(legends[TOSIdx])
            if subtitles is not None:
                if 'train' in subtitles[sliceIdx]:
                    subtitleColor = titleColors[0]
                else:
                    subtitleColor = titleColors[1]
                # print(subtitleColor)
                axe.set_title(subtitles[sliceIdx], color = subtitleColor)
            axe.set_yticks(np.arange(1,19)-0.5)
            axe.set_yticklabels(np.arange(1,19))
        axe.legend()
    for sliceIdx in range(NSlices, NRows*NCols):
        axe = axs[sliceIdx]
        axe.axis("off")
    fig.savefig(save_filename, bbox_inches='tight')   # save the figure to file
    plt.close(fig)

def saveArraysFig2File(arrays: list, save_filename:str, legends = None, hist = False, title = None):
    import seaborn as sns
    plt.ioff()
    plt.style.use('seaborn')
    fig, axe = plt.subplots()
    axe.set_title(title) if title is not None else None
    for arrayIdx,array in enumerate(arrays):
        if hist:
            sns.distplot(array, label = legends[arrayIdx] if legends is not None else None, bins = 20)
            # axe.hist(array, label = legends[arrayIdx] if legends is not None else None, bins = 20, alpha = 0.5)
        else:
            line, = axe.plot(array)            
            line.set_label(legends[arrayIdx]) if legends is not None else None
    axe.legend()
    fig.savefig(save_filename, bbox_inches='tight')   # save the figure to file
    plt.close(fig)

def saveLossInfo(pred, label, savePath):
    getDataNormArr = lambda data: np.sqrt(np.sum((data)**2, axis = tuple(range(1,np.ndim(data)))).flatten())
    normL2Pred = getDataNormArr(pred)
    normL2GT   = getDataNormArr(label)    
    lossL2     = getDataNormArr(pred - label)
    saveArraysFig2File([normL2Pred, normL2GT], savePath + 'norml2.png', ['predTe','GTTe'], title = 'Data L2 Norm')
    saveArraysFig2File([normL2Pred, normL2GT], savePath + 'norml2hist.png', ['predTe', 'GTTe'], title = 'Data L2 Norm Histogram', hist = True)
    saveArraysFig2File([lossL2], savePath + 'lossl2.png', ['test'], title = 'Data L2 Loss')
    saveArraysFig2File([lossL2], savePath + 'lossl2hist.png', ['test'], hist=True, title = 'Data L2 Loss Histogram')
    
    # saveArraysFig2File([lossL2Tr, lossL2Te], savePath + 'lossl2hist.png', ['train', 'test'], hist=True, title = 'Data L2 Loss Histogram')
    # if dataConfig.get('mergeAllSlices', False):
    #     getDataNormArr4Merged = lambda data: np.linalg.norm(data.reshape(-1,data.shape[-1]*data.shape[-2]), axis = 1)
    #     # lossL2BySliceTr = np.linalg.norm((predTrRaw - labelDataTrRaw).reshape(-1,predTrRaw.shape[-1]*predTrRaw.shape[-2]), axis = 1) # [N, NSlices, 1, NSectors] -> [N*NSlices, NSectors]
    #     normL2PredBySliceTe = getDataNormArr4Merged(predTeRaw)
    #     normL2GTBySliceTe = getDataNormArr4Merged(labelDataTeRaw)
    #     lossL2BySliceTe = getDataNormArr4Merged(predTeRaw - labelDataTeRaw)        
        
    #     #lossL2BySliceTr = np.linalg.norm(predTrRaw - labelDataTrRaw, axis = 0).flatten()
    #     #lossL2BySliceTe = np.linalg.norm(predTeRaw - labelDataTeRaw, axis = 0).flatten()
    #     saveArraysFig2File([normL2PredBySliceTr, normL2PredBySliceTe, normL2GTBySliceTr, normL2GTBySliceTe], savePath + 'norml2BySlice.png', ['predTr', 'predTe', 'GTTr', 'GTTe'], title = 'Slice L2 Norm')
    #     saveArraysFig2File([normL2PredBySliceTr, normL2PredBySliceTe, normL2GTBySliceTr, normL2GTBySliceTe], savePath + 'norml2histBySlice.png', ['predTr', 'predTe', 'GTTr', 'GTTe'], title = 'Slice L2 Norm Histogram', hist = True)
    #     saveArraysFig2File([lossL2BySliceTr, lossL2BySliceTe], savePath + 'lossl2ByPatient.png', ['train', 'test'], title = 'Slices L2 Loss')
    #     saveArraysFig2File([lossL2BySliceTr, lossL2BySliceTe], savePath + 'lossl2histByPatient.png', ['train', 'test'], hist=True, title = 'Slices L2 loss Histogram')

def safeLoadMedicalImg(filename, read_dim = 0):
    import nibabel as nib
    import SimpleITK as sitk
    dataFormat = filename.split('.')[-1]
    if dataFormat.lower() == 'nii':
        img = nib.load(filename).get_fdata()
    elif dataFormat.lower() == 'mhd':
        # Strangely, [80, 70, 50] image become [50, 80, 70]!
        # i.e. [H, W, N] => [N, H, W]
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        img = np.rollaxis(img, 0, 3)
    
    if read_dim != 2:
        img = np.rollaxis(img, read_dim, 2)
            
    return img


def convertTensorformat(img, sourceFormat, targetFormat, targetDim=0, sourceSliceDim=0):
    '''
    @description:
        Convert ime tensor format among 3D medical images, Tensorflow and Pytorch
    '''
    # 1. Convert from source format to tensorflow in same dimension
    if sourceFormat.lower() == 'single3dgrayscale':
        # [D1, D2, D3] -> [1, D, H, W, 1]
        img = np.rollaxis(img, sourceSliceDim)
        img = img[np.newaxis, :, :, :, np.newaxis]
        # img = np.expand_dims(img, -1)
    elif sourceFormat.lower() == 'tensorflow':
        pass
    elif sourceFormat.lower() == 'pytorch':
        img =  np.moveaxis(img, 1, -1)
#        img = np.moveaxis(img, -1, 1)
    
    sourceDim = len(np.shape(img)) - 2
    if targetDim == 0:
        targetDim = sourceDim
        
    # 2. Convert to target format
    if targetFormat != 'tensorflow':
        return convertTFTensorTo(img, targetFormat, targetDim)
    else:        
        if sourceDim == targetDim:
            return img
        elif sourceDim == 2:
            # [N, H, W, C] -> [N, D, H, W, C]
            return img[:,np.newaxis,:,:,:]
        elif sourceDim == 3:
            # [N, D, H, W, C] -> [N, H, W, C]            
            if np.shape(img)[1] != 1:
                raise ValueError(f'Cannot convert')
            else:
                return img[:, 0, :, :, :]

def convertTFTensorTo(img, targetFormat, targetDim):
    '''    
    @description: 
        Convert tensorflow tensor format into 3D grayscale or Pyrotch format tensors
    @params:
        img{numpy ndarray}:
            tensorflow tensor with shape [N, H, W, C] or [N, D, H, W, C]
        lang{string}:
            single3DGrayscale or Pyrotch.
            single3DGrayscale:
                2D: [D, H, W]
            Pyrotch:
                2D: [N, C, H, W]
                3D: [N, C, T, H, W]
        dim{int}:
            Treate the 3D image as stack of 2D or 3D tensors
    @return:         
    '''
    sourceDim = len(np.shape(img)) - 2
    
    if targetFormat.lower() == 'single3dgrayscale':
        # [N, H, W, C] or [N, D, H, W, C] -> [D, H, W]        
        if sourceDim == 2:
            # [N, H, W, C] -> ?
            raise ValueError('Not supported')
        elif sourceDim == 3:
            # [N, D, H, W, C] -> [D, H, W]
            if np.shape(img)[0] == 1 and np.shape(img)[-1] == 1:
                return img[0,:,:,:,0]
            else:
                raise ValueError('Not supported')
            
    elif targetFormat.lower() == 'pytorch':
        if sourceDim == targetDim:
            # [N, D, H, W, C] -> [N, C, D, H, W]
            # [N, H, W, C] -> [N, C, H, W]
            return np.moveaxis(img, -1, 1)
        elif sourceDim == 2:
            # [N, H, W, C] -> [N, C, H, W] -> [N, C, D, H, W]
            img = np.moveaxis(img, -1, 1)
            return img[:,:,np.newaxis,:,:]
        elif sourceDim == 3:
            # [N, D, H, W, C] -> [N, H, W, C] -> [N, C, H, W]
            if np.shape(img)[1] != 0:
                raise ValueError('Not supported')
            else:
                img = img[:,0,:,:]
                return np.moveaxis(img, -1, 1)
    
def convertMedicalTensorTo(img, lang, dim, sliceDim = 0):
    '''    
    @description: 
        Convert 3D medical image (with dim X, Y, Z) into Tensorflow or Pyrotch format tensors
    @params:
        img{numpy ndarray}:
            3D medical image with shape [X, Y, Z]
        lang{string}:
            Tensorflow or Pyrotch.
            Tensorflow:
                2D: [N, H, W, C]
                3D: [N, D, H, W, C]
            Pyrotch:
                2D: [N, C, H, W]
                3D: [N, C, T, H, W]
        dim{int}:
            Treate the 3D image as stack of 2D or 3D tensors
        sliceDim{int}:
            if dim == 2, then sliceDim refer to teh dim to look into (slice)
    @return:         
    '''
    # 1. Convert image into [N, H, W]
    img = np.rollaxis(img, sliceDim)
    
    # 2. Convert to target format
    if dim == 2 and lang.lower() == 'tensorflow':
        img = img[:, :, :, np.newaxis]
    elif dim == 2 and lang.lower() == 'pytorch':
        img = img[:, np.newaxis, :, :]
    elif dim == 3 and lang.lower() == 'tensorflow':
        img = img[np.newaxis, :, :, :, np.newaxis]
    elif dim == 3 and lang.lower() == 'pytorch':
        img = img[:, np.newaxis, np.newaxis, :, :]
    else:
        raise ValueError(f"Unsupported combination: target format {lang} with dim {dim}")
    return img

def safeDivide(a,b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def createExpFolder(resultPath, configName, create_subFolder = False, addDate = True):
    # https://datatofish.com/python-current-date/
    import os
    import datetime
    if addDate:
        #today = datetime.date.today().strftime('%d-%b-%Y')
        today = datetime.date.today().strftime('%Y-%b-%d')
        expName = today + '-' + configName
    else:
        expName = configName
    expPath = resultPath + '/' + expName + '/'
    if not os.path.exists(expPath):
        os.mkdir(expPath)
    else:
        for idx in range(2,100):
            if addDate:
                expName = today + '-' + configName + '-' + str(idx)
            else:
                expName = configName + '-' + str(idx)
            expPath = resultPath + '/' + expName + '/'
            if not os.path.exists(expPath):
                os.mkdir(expPath)
                break
    if create_subFolder:
        os.mkdir(expPath + '/valid_img')
        os.mkdir(expPath + '/train_img')
        os.mkdir(expPath + '/checkpoint')

    return expPath, expName


def getConfigFromFile(jsonFilename):
    import json
    with open(jsonFilename, 'r') as f:
        config = json.load(f)
    return config

def saveConfig2Json(config, jsonFilename):
    # https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    import json
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                # print(obj)
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NpEncoder, self).default(obj)    
    
    with open(jsonFilename, 'w') as f:
        json.dump(config, f, indent=4, sort_keys=False, cls=NpEncoder)
