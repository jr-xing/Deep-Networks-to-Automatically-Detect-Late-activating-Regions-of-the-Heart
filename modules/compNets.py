# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:37:02 2020

@author: remus
"""

import copy
from torch import nn
class NetComp(nn.Module):
    def __init__(self, netPretrained, netFinetune ,config = {}):
        super(NetComp, self).__init__()
        self.netPretrained = copy.deepcopy(netPretrained)
        self.netFinetune = copy.deepcopy(netFinetune)
            
        # Freeze Weights
        # https://discuss.pytorch.org/t/freeze-the-learnable-parameters-of-resnet-and-attach-it-to-a-new-network/949
        for param in self.netPretrained.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.netFinetune(self.netPretrained.forward_one(x))