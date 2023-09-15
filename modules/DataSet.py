#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:55:43 2019

@author: jrxing
"""

import random
import torch
import numpy as np
from torch.utils.data import Dataset
# from torchvision import transforms
# from utils.io import safeLoadMedicalImg
class DataSet2D(Dataset):
    """
    Load in 3D medical image, treate image as a stack of 2D images with given dimension
    """
    def __init__(self, imgs, labels, labelmasks = None, transform = None, device = torch.device("cpu")):
        super(DataSet2D, self).__init__()
        # img should be [N,H,W,C] or [N, T, H, W, C]
        data = imgs
        
        # Note that in transform pytorch assume the image is PILImage [H, W, C]!
        self.data = data
        self.labels = labels
        if labelmasks is None:
            #labelmasks = ['None'] * len(data)
            # labelmasks = np.ones(len(data)) * np.nan
            # labelmasks = np.ones(len(data))
            labelmasks = np.ones(labels.shape)
        self.labelmasks = labelmasks
        self.transform= transform
#        if transform != None:
#            self.data = self.transform(self.data)
#        self.data.to(dtype=torch.float)
        self.dataShape = self.data.shape
    
    def __len__(self):
        return np.shape(self.data)[0]
    
    def __getitem__(self, idx):
        sample = self.data[idx, :].astype(np.float32)
        label  = self.labels[idx]
        labelmask   = self.labelmasks[idx]
        if self.transform is not None:
#            sample = transforms.ToPILImage()(sample)
            sample = self.transform(sample)        
        # return sample
        return {'data': sample, 'label': label, 'labelmask': labelmask}


class DataSetContrastive(Dataset):
    def __init__(self, data: np.ndarray, dataSourceIDs: np.ndarray, transform = None, device = torch.device("cpu")):
        super(DataSetContrastive, self).__init__()
        self.data = data
        self.dataSourceIDs = dataSourceIDs
        self.dataSourceUniqueIDs = np.unique(dataSourceIDs)
        self.N = data.shape[0]
        self.NUniqueIDs = len(self.dataSourceUniqueIDs)
        
    
    def __getitem__(self, idx):
        useSameCls = np.random.randint(0, 2)
        
        if useSameCls:
            cls2UseIdx = np.random.randint(0, self.NUniqueIDs)
            # print(np.where(self.dataSourceIDs == self.dataSourceUniqueIDs[cls2UseIdx]))
            dataIndicesInCls = np.where(self.dataSourceIDs == self.dataSourceUniqueIDs[cls2UseIdx])[0]
            dataPairIndicesInCls = np.array(random.sample(list(dataIndicesInCls),2))
            data0Idx = dataPairIndicesInCls[0]
            data1Idx = dataPairIndicesInCls[1]
        else:
            dataClses = random.sample(range(0, self.NUniqueIDs), 2)
            data0Cls  = dataClses[0]
            data0Idx  = random.sample(list(np.where(self.dataSourceIDs==data0Cls)[0]), 1)[0]
            data1Cls  = dataClses[1]
            data1Idx  = random.sample(list(np.where(self.dataSourceIDs==data1Cls)[0]), 1)[0]            
        
        return {'data0': self.data[data0Idx], \
                'data1': self.data[data1Idx], \
                'label': useSameCls}
            
    def __len__(self):
        return self.N
    
if __name__ == '__main__':
    NCls = 5
    data = np.zeros((NCls*5, 1, 32, 32))
    dataSourceIDs = np.zeros(NCls*5)
    for clsIdx in range(NCls):
        data[clsIdx*5:(clsIdx+1)*5] = clsIdx
        dataSourceIDs[clsIdx*5:(clsIdx+1)*5] = clsIdx
        
    dataset = DataSetContrastive(data, dataSourceIDs)
    data0 = dataset[0]
    print(data0['data0'].shape)