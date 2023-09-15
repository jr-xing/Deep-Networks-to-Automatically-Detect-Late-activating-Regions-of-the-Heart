# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:04:18 2020

@author: remus
"""

import torch
import numpy as np
from torch import nn

class NetSiamese(nn.Module):
    # https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
    # https://github.com/fangpin/siamese-pytorch/blob/master/model.py
    def __init__(self, config = {}):
        super(NetSiamese, self).__init__()
        
        self.paras = config.get('paras', {})
        self.dim = self.paras.get('dim', 2)
        self.n_input_channels = self.paras.get('n_input_channels', 1)
        self.n_sectors = self.paras.get('n_sectors', 18)
        self.n_frames  = self.paras.get('n_frames',  25)
        self.n_conv_layers = self.paras.get('n_conv_layers',  4)
        self.n_conv_channels = self.paras.get('n_conv_channels',  8)
        self.n_feature_channels = self.paras.get('n_feature_channels',  8)
        self.conv_size = self.paras.get('conv_size',  3)
        H = self.n_sectors
        W = self.n_frames
        T = self.paras.get('T', 1)
        
        def netf_block_layers(n_input_channels, n_output_channels, conv_size, stride = 1, padding = 1, batchNorm = False, pooling = False, dim = 2):
            if dim == 2:
                layers = [nn.Conv2d(n_input_channels, n_output_channels, conv_size, stride=1, padding=1), \
                          nn.ReLU(True)]
                if batchNorm:
                    layers.append(nn.BatchNorm2d(n_output_channels))
                if pooling:
                    layers.append(nn.MaxPool2d(kernel_size = (1,2)))
            elif dim == 3:
                layers = [nn.Conv3d(n_input_channels, n_output_channels, conv_size, stride=1, padding=1), \
                          nn.ReLU(True)]
                if batchNorm:
                    layers.append(nn.BatchNorm3d(n_output_channels))
                if pooling:
                    layers.append(nn.MaxPool3d(kernel_size = (1,1,2)))
            return layers
        
        # Feature net
        pooling = self.paras.get('pooling', False)
        batchNorm = self.paras.get('batchNorm', True)        
        self.netf_conv_layers = netf_block_layers(self.n_input_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1, batchNorm = batchNorm, pooling = pooling)
        for midLayerIdx in range(self.n_conv_layers-2):            
            self.netf_conv_layers += netf_block_layers(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1, batchNorm = batchNorm, pooling = pooling)
        self.netf_conv_layers += netf_block_layers(self.n_conv_channels, self.n_feature_channels, self.conv_size, stride=1, padding=1, batchNorm = batchNorm, pooling = pooling)
        
        decreaseTimeByPooling = 2**self.n_conv_layers if pooling else 1
        
        linearInputSize = H*(W//decreaseTimeByPooling)*T*self.n_feature_channels        
        self.netf_linear_layers = \
            [nn.Flatten()] + \
            [nn.Linear(linearInputSize, linearInputSize//2), nn.ReLU(True)] +\
            [nn.Linear(linearInputSize//2, linearInputSize//2), nn.ReLU(True)] +\
            [nn.Linear(linearInputSize//2, 100), nn.ReLU(True)]
        self.netf = nn.ModuleList(self.netf_conv_layers + self.netf_linear_layers)
            
        # Surrogate net
        # self.nets_layers = nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 1), nn.Sigmoid())
        # self.nets = nn.ModuleList(self.nets_layers)
        
    def forward_one(self, x):
        for layer in self.netf:
            # print(x.shape)
            # print(layer)
            x = layer(x)
        # x = nn.Flatten()(x)
        # for layer in self.nets:
        #     # print(x.shape)
        #     x = layer(x)
        return x
    
    def forward(self, x1, x2):
        feature1 = self.forward_one(x1)
        feature2 = self.forward_one(x2)        
        return feature1, feature2

class NetSimpleFCN(nn.Module):
    def __init__(self, config = {}):
        super(NetSimpleFCN, self).__init__()
        self.inputDim = config.get('inputDim', 100)
        self.outputDim = config.get('outputDim', 18)
        
        self.net = nn.Sequential(nn.Linear(self.inputDim, 50),nn.Linear(50, 50), nn.Linear(50, self.outputDim), nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        x = nn.LeakyReLU(inplace = True)(x-17)+17
        return x

if __name__ == '__main__':
    H = 18; W = 60
    dataTest0 = torch.rand(10, 1, 18, 59)
    dataTest1 = torch.rand(10, 1, 18, 59)
    net_configTest = {'paras':{'n_sectors':H, 'n_frames':W, 'pooling': True, 'batchNorm': True}}
    netTest = NetSiamese(net_configTest)
    
    outputTest_one = netTest.forward_one(dataTest0)
    outputTest0, outputTest1 = netTest(dataTest0, dataTest1)
    
    print(outputTest0.shape)
    
    netTOSTest = NetSimpleFCN()
    TOSTest = netTOSTest(outputTest0)
    print(TOSTest)
