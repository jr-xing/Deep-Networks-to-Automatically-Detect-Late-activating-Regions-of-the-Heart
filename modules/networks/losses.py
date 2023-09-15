# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:07:17 2020

@author: remus
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def get_loss(config):
    # https://blog.csdn.net/gwplovekimi/article/details/85337689
    name = config['name']
    para = config['para']
    if name == 'MSE':
        return nn.MSELoss()
    elif name =='MSE_POWER':
        return MSE_POWER(power = para.get('power',2))
    elif name == 'TV2D':
        return TVLoss2D(para.get('TV_weight', 1))
    elif name == 'TV3D':
        return TVLoss3D(para.get('TV_weight', 1))
    elif name == 'strainMat1D':
        return nn.MSELoss(reduction='sum')
        # return TOSMSE()
        # return lambda pred, GT: nn.MSELoss()(pred, GT)
        # return lambda pred, GT: nn.MSELoss()(pred, GT) + \
            # 0 * TVLoss1D()(pred, GT) + \
            # 2 * torch.norm(pred[pred<0], p=2)
    elif name == 'strainMatSeg':
        return lambda pred, GT: nn.CrossEntropyLoss()(pred, torch.squeeze(GT)) + \
            0 * TVLoss2D(WX = 0, WY = 1)(pred, GT)
    elif name == 'loss3':
        return Loss3()
    elif name == 'contrastive':
        return ContrastiveLoss()
    else:
        raise ValueError(f'Unsupported loss type: {name}')

def slice_img(img, config):
    if len(np.shape(img)) == 4:
        # if images are 2D image and img has shape [N,C,H,W]
        img_sample = img[config.get('index', 0), :]            
    elif len(np.shape(img)) == 5:
        # if images are 3D image and img has shape [N,C,D,H,W]
        img_sample_3D = img[config.get('index', 0), :]
#        print(np.shape(img_sample_3D))
        slice_axis = config.get('slice_axis',2)
        slice_index = config.get('slice_index',0)
        if slice_index == 'middle': slice_index = int(np.shape(img_sample_3D)[slice_axis]/2)
        if slice_axis == 0:             
            img_sample = img_sample_3D[:,slice_index,:,:]
        elif slice_axis == 1:
            img_sample = img_sample_3D[:,:,slice_index,:]
        elif slice_axis == 2:
            img_sample = img_sample_3D[:,:,:,slice_index]
    else:
        raise ValueError(f'Wrong image dimension. \
                         Should be 4 ([N,C,H,W]) for 2d images \
                         and 5 ([N,C,D,H,W]) for 3D images, \
                         but got {len(np.shape(img))}')
    return img_sample

import torch.nn as nn
class Loss3(nn.Module):
    def __init__(self):
        super(Loss3,self).__init__()
    def forward(self, output, truth):
        lossLoc, lossTime, lossDetail = 0, 0, 0
        N, NSectors = output.shape
        
        for dataIdx in range(N):
            # https://stackoverflow.com/questions/54969646/how-does-pytorch-backprop-through-argmax
            outputDatum, truthDatum = output[dataIdx].reshape(1,-1), truth[dataIdx].reshape(1,-1)
            # 1. Location Loss
            outputShiftedMat = torch.cat([torch.roll(outputDatum, shift) for shift in range(NSectors)])
            shiftDiff = torch.norm(outputShiftedMat - truthDatum, dim = 1)
            # bestShift = torch.argmin(shiftDiff)*1.0
            bestShift = torch.min(shiftDiff,0)[1]#*1.0
            lossLoc += bestShift  # about 0~5
            outputDatum = torch.roll(outputDatum, int(bestShift))
            
            # 2. Time Loss            
            outputPeak = torch.max(outputDatum)
            truthPeak = torch.max(truthDatum)
            bestScale = truthPeak / outputPeak
            lossTime += torch.abs(bestScale - 1)  # about 0~0.5            
            outputDatum = outputDatum*bestScale
            
            # 3. Detail / Residule Loss
            outputGrad = ((outputDatum - torch.roll(outputDatum, 1)) + (outputDatum - torch.roll(outputDatum, -1)))/2
            truchGrad = ((truthDatum - torch.roll(truthDatum, 1)) + (truthDatum - torch.roll(truthDatum, -1)))/2            
            lossDetail += torch.norm(outputGrad - truchGrad)  # About 30
            
        # 4. MSE
        lossMSE = torch.norm(output - truth)
            
        # print(lossLoc.dtype, lossTime.dtype, lossDetail.dtype)
        lossTotal = 50*lossLoc + 100*lossTime + 0.1*lossDetail + lossMSE
        return lossTotal
        
    
    def forward_numpy(self, output, truth):
        # Output shape: [N, NSectors]
        lossLoc, lossTime, lossDetail = 0, 0, 0
        N, NSectors = output.shape
        
        for dataIdx in range(N):
            outputDatum, truthDatum = output[dataIdx].reshape(1,-1), truth[dataIdx].reshape(1,-1)
            # 1. Location Loss
            outputShiftedMat = np.concatenate([np.roll(outputDatum, shift) for shift in range(NSectors)])
            shiftDiff = np.linalg.norm(outputShiftedMat - truthDatum, axis = 1)
            bestShift = np.argmin(shiftDiff)
            lossLoc += bestShift  # about 0~5
            outputDatum = np.roll(outputDatum, bestShift)
            
            # 2. Time Loss            
            outputPeak = np.max(outputDatum)
            truthPeak = np.max(truthDatum)
            bestScale = truthPeak / outputPeak
            lossTime += np.abs(bestScale - 1)  # about 0~0.5            
            outputDatum = outputDatum*bestScale            
            
            # 3. Detail / Residule Loss
            lossDetail += np.linalg.norm(outputDatum - truthDatum)  # About 30
        
        # print(torch.norm(output-truth))
        # print(torch.norm(output[output<17]))
        lossTotal = lossLoc + 10*lossTime + lossDetail
        return lossTotal



class TOSMSE(nn.Module):
    def __init__(self):
        super(TOSMSE,self).__init__()
    
    def forward(self, output, truth):
        print(torch.norm(output-truth))
        # print(torch.norm(output[output<17]))
        # return torch.norm(output-truth) + 0.05*torch.norm(output[output<17])

class MSE_POWER(nn.Module):
    def __init__(self, power = 2):
        super(MSE_POWER,self).__init__()
        self.power = power
    
    def forward(self, output, truth):
        return torch.norm(output**self.power-truth**self.power)

class TVLoss1D(nn.Module):
    # def __init__(self,TVLoss_weight=1, MSELoss_weight = 1):
    def __init__(self):
        super(TVLoss1D,self).__init__()
        #self.TVLoss_weight = TVLoss_weight
        # self.MSELoss_weight = MSELoss_weight
 
    def forward(self, output, truth):
        # [N, C, H, W]
        # [N, C, T, H, W]
        batch_size = output.size()[0]
        tv = torch.pow((output[:,1:]-output[:,:-1]),2).sum()
        
        # l2_loss = torch.pow(output - truth, 2).sum() / (self._tensor_size(output)*batch_size)
        tv_loss = tv/batch_size
        return tv_loss
        # return self.MSELoss_weight * l2_loss + self.TVLoss_weight * tv_loss
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class TVLoss2D(nn.Module):
    # def __init__(self,TVLoss_weight=1, MSELoss_weight = 1):
    def __init__(self, WY = 1, WX = 1):
        super(TVLoss2D,self).__init__()
        self.WY = WY
        self.WX = WX
        #self.TVLoss_weight = TVLoss_weight
        # self.MSELoss_weight = MSELoss_weight
 
    def forward(self, output, truth):
        # [N, C, H, W]
        # [N, C, T, H, W]
        batch_size = output.size()[0]
        h_x = output.size()[2]
        w_x = output.size()[3]
        count_h = self._tensor_size(output[:,:,1:,:])
        count_w = self._tensor_size(output[:,:,:,1:])
        h_tv = torch.pow((output[:,:,1:,:]-output[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((output[:,:,:,1:]-output[:,:,:,:w_x-1]),2).sum()
        
        tv_loss = (self.WY * h_tv/count_h + self.WX * w_tv/count_w)/batch_size
        return tv_loss
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
# import torch.nn as nn
class TVLoss3D(nn.Module):
    def __init__(self,TV_weight=1):
        super(TVLoss3D,self).__init__()
        self.TV_weight = TV_weight
 
    def forward(self, output, truth):
        # [N, C, H, W]
        # [N, C, T, H, W]
        N, C, D, H, W = output.shape
        count_d = self._tensor_size(output[:,:,1:,:,:])
        count_h = self._tensor_size(output[:,:,:,1:,:])
        count_w = self._tensor_size(output[:,:,:,:,1:])
        
        d_tv = torch.pow((output[:,1:,:,:]-output[:,:,:D-1,:,:]),2).sum()
        h_tv = torch.pow((output[:,:,1:,:]-output[:,:,:,:H-1,:]),2).sum()
        w_tv = torch.pow((output[:,:,:,1:]-output[:,:,:,:,:W-1]),2).sum()
        
        l2_loss = torch.pow(output - truth, 2).sum() / (self._tensor_size(output)*N)
        tv_loss = (d_tv/count_d + h_tv/count_h + w_tv/count_w)/N
        return l2_loss + self.TV_weight * tv_loss
#        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    From: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive