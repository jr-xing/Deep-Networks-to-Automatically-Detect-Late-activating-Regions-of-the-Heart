# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:49:14 2020

@author: remus
"""
import torch
import numpy as np
from torch import nn
from modules.networks.Siamese import NetSiamese, NetSimpleFCN
def getNetwork(config):
    if config['type'] in ['Siamese']:
        net = NetSiamese(config)
    elif config['type'] in ['simpleFCN']:
        net = NetSimpleFCN(config)
    else:
        if config['inputType'] in ['strainMat', 'strainMatSVD', 'dispFieldJacoMat', 'strainMatFullResolution']:
            if not config.get('mergeAllSlices', False):
                net = NetStrainMat2TOS(config)
            else:
                net = NetMultiSliceStrainMat2TOS(config)
        elif config['inputType'] == 'strainMatSub':
            net = NetSubStrainMat2TOS(config)
        elif config['inputType'] in ['dispField', 'dispFieldFv']:
            if config['outputType'] == 'TOS':
                _, _, T, H, W = config['inputShape']
                config['paras']['H'] = H
                config['paras']['W'] = W
                config['paras']['T'] = T
                config['n_input_channels'] = 2
                net = NetStrainImg2TOS(config)
            elif config['outputType'] == 'TOSImage':
                net = NetStrainImg2TOSImg(config)
            else:
                raise ValueError('NOT SUPPORT')
        elif config['inputType'] == 'strainImg':
            config['paras']['C'] = 1
            net = NetStrainImg2TOS(config)
        elif config['inputType'] == 'strainCurve':
            net = NetStrainCurve2TOS(config)
        else:
            raise ValueError(f'Unrecognized data type {config["inputType"]}')
    return net



class NetStrainCurve2TOS(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainCurve2TOS, self).__init__()
        # self.imgDim = 3
        self.paras = config.get('paras', None)
        self.n_sectors = self.paras.get('n_sectors', 18)
        self.n_frames  = self.paras.get('n_frames',  25)
        self.n_mid_linear_layers = self.paras.get('n_mid_linear_layers',  3)
        self.mid_linear_width = self.paras.get('mid_linear_width',  60)
        layers = [nn.Linear(self.n_frames, self.mid_linear_width), nn.ReLU(True)] + \
                        [nn.Linear(self.mid_linear_width, self.mid_linear_width), nn.ReLU(True)] * (self.n_mid_linear_layers-2) + \
                        [nn.Linear(self.mid_linear_width, 1)]        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x): 
        for layer in self.layers:
            x = layer(x)        
        return x

class NetStrainMat2TOS(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainMat2TOS, self).__init__()
        # self.imgDim = 3
        self.paras = config.get('paras', None)
        self.n_input_channels = self.paras.get('n_input_channels', 1)
        self.n_sectors = self.paras.get('n_sectors', 18)
        self.n_frames  = self.paras.get('n_frames',  25)
        self.n_conv_layers = self.paras.get('n_conv_layers',  4)
        self.n_conv_channels = self.paras.get('n_conv_channels',  16)
        self.conv_size = self.paras.get('conv_size',  3)
        # print(self.n_conv_layers)
        # print('HAHA')        　
        # convs = [nn.Conv2d(self.n_input_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] + \
        #                 [nn.Conv2d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] * (self.n_conv_layers-2) + \
        #                 [nn.Conv2d(self.n_conv_channels, 1, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        
        convs = [nn.Conv2d(self.n_input_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        for innerLayerIdx in range(self.n_conv_layers-2):
            convs += [nn.Conv2d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        convs += [nn.Conv2d(self.n_conv_channels, 1, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        
        
        linear = [nn.Flatten(), nn.Linear(self.n_sectors*self.n_frames, self.n_sectors)]
        self.layers = nn.ModuleList(convs+linear)
        
    def forward(self, x):
        # x = self.net(x)        
        for layer in self.layers:
            x = layer(x)        
        x = nn.LeakyReLU(inplace = True)(x-17)+17
        return x

class NetMultiSliceStrainMat2TOS(nn.Module):
    def __init__(self, config = {}):
        super(NetMultiSliceStrainMat2TOS, self).__init__()
        # [N, 1, NSlices, NSectors, NFrames] 
        # -> (rollaxis) [N, NFrames, NSlices, NSectors]
        # -> (conv2d) [N, 1, NSlices, NSectors]        
        # -> (reshape to label) [N, NSlices, 1, NSectors]
        self.paras = config.get('paras', None)
        self.n_input_channels = self.paras.get('n_input_channels', 1)
        self.n_sectors = self.paras.get('n_sectors', 18)
        self.n_frames  = self.paras.get('n_frames',  25)
        self.n_slices = self.paras.get('n_slices', 9)
        self.n_conv3d_layers = self.paras.get('n_conv3d_layers',  4)
        self.n_conv3d_channels = self.paras.get('n_conv3d_channels',  16)
        self.n_conv2d_layers = self.paras.get('n_conv2d_layers',  1)
        self.n_conv2d_channels = self.paras.get('n_conv2d_channels',  8)
        
        self.conv_size = self.paras.get('conv_size',  3)
        if self.n_conv3d_layers == 1:
            conv3Ds = [nn.Conv3d(self.n_input_channels, 1, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        else:
            # conv3Ds = [nn.Conv3d(self.n_input_channels, self.n_conv3d_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] + \
            #                 [nn.Conv3d(self.n_conv3d_channels, self.n_conv3d_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] * (self.n_conv3d_layers-2) + \
            #                 [nn.Conv3d(self.n_conv3d_channels, 1, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
            conv3Ds = [nn.Conv3d(self.n_input_channels, self.n_conv3d_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
            for innerLayerIdx in range(self.n_conv3d_layers-2):
                conv3Ds += [nn.Conv3d(self.n_conv3d_channels, self.n_conv3d_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
            conv3Ds += [nn.Conv3d(self.n_conv3d_channels, 1, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        
        if self.n_conv2d_layers == 1:
            conv2Ds = [nn.Conv2d(self.n_frames, 1, self.conv_size, stride = 1, padding = 1)]
        else:
            # conv2Ds = [nn.Conv2d(self.n_frames, self.n_conv2d_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] + \
            #                 [nn.Conv2d(self.n_conv2d_channels, self.n_conv2d_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] * (self.n_conv2d_layers-2) + \
            #                 [nn.Conv2d(self.n_conv2d_channels, 1, self.conv_size, stride=1, padding=1)]
                            
            conv2Ds = [nn.Conv2d(self.n_frames, self.n_conv2d_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
            for innerLayerIdx in range(self.n_conv2d_layers-2):
                conv2Ds += [nn.Conv2d(self.n_conv2d_channels, self.n_conv2d_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
            conv2Ds += [nn.Conv2d(self.n_conv2d_channels, 1, self.conv_size, stride=1, padding=1)]
            
        self.conv3Ds = nn.ModuleList(conv3Ds)
        self.conv2Ds = nn.ModuleList(conv2Ds)        
        # self.conv2D = nn.Conv2d(self.n_frames, 1, self.conv_size, stride = 1, padding = 1)
        
        
    def forward(self, x):
        # 1. Convolution
        #   [N, 1, NSlices, NSectors, NFrames]  -> [N, 1, NSlices, NSectors, NFrames] 
        for layer in self.conv3Ds:
            x = layer(x)
        
        # 2. Roll and squeeze axis to fit Conv2d (in order to "squeeze" frame dim to 1)
        #   [N, 1, NSlices, NSectors, NFrames] 
        #   -> [N, NSlices, NSectors, NFrames]
        #   -> [N, NFrames, NSlices, NSectors]
        #   -> [N, 1, NSlices, NSectors]
        x = x[:,0,:,:,:]
        x = x.permute(0,3,1,2)
        for layer in self.conv2Ds:
            x = layer(x)
        # x = self.conv2D(x)
        
        # 3. Roll axis to fit label
        #   [N, 1, NSlices, NSectors] -> [N, NSlices, 1, NSectors]
        x = x.permute(0,2,1,3)
        
        # 4. Make back to 3d to fit input dimension
        # [N, NSlices, 1, NSectors] -> [N, 1, NSlices, 1, NSectors]
        x = x[:,None,:,:,:]
        
        # 5. Remove "negative" values
        x = nn.LeakyReLU(inplace = True)(x-17)+17
        return x

class NetSubStrainMat2TOS(nn.Module):
    def __init__(self, config = {}):
        super(NetSubStrainMat2TOS, self).__init__()
        # self.imgDim = 3
        self.paras = config.get('paras', {})
        self.n_sectors = self.paras.get('n_sectors', 18)
        self.n_frames  = self.paras.get('n_frames',  25)
        self.n_conv_layers = self.paras.get('n_conv_layers',  4)
        self.n_conv_channels = self.paras.get('n_conv_channels',  16)
        self.conv_size = self.paras.get('conv_size',  3)
        self.division_width = self.paras.get('division_width',  5)
        # convs = [nn.Conv2d(1, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] + \
        #                 [nn.Conv2d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] * (self.n_conv_layers-2) + \
        #                 [nn.Conv2d(self.n_conv_channels, 1, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
                        
        convs = [nn.Conv2d(1, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        for midLayerIdx in range(self.n_conv_layers-2):
            convs += [nn.Conv2d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        convs += [nn.Conv2d(self.n_conv_channels, 1, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        
        
        linear = [nn.Flatten(), nn.Linear(self.division_width*self.n_frames, 1)]        
        self.layers1 = nn.ModuleList(convs+linear)
        # self.linear = nn.ModuleList(linear)
        self.layers2 = nn.ModuleList([nn.Linear(self.n_sectors, self.n_sectors)])
        
        
    def forward(self, x):
        # x = self.net(x)
        # print('forward')
        half_width = self.division_width // 2
        layer1_outputs = [np.nan]*self.n_sectors
        for sectorIdx in range(self.n_sectors):
            # print(np.arange(sectorIdx-half_width,sectorIdx+half_width+1))
            smin = sectorIdx-half_width
            smax = sectorIdx+half_width
            subx = x[:,:,torch.arange(sectorIdx-half_width,sectorIdx+half_width+1)%self.n_sectors,:]
            # print('subX:',subx.shape)
            for layer in self.layers1:
                # print(subx.shape, '->', layer)
                subx = layer(subx)
            # print(subx.shape)
            # subx = self.linear(subx)
            layer1_outputs[sectorIdx] = subx
        layer1_output = torch.cat(layer1_outputs, axis=1)
        # print('layer1_output: ',layer1_output.shape)
        
        for layer in self.layers2:
            layer1_output = layer(layer1_output)
        return layer1_output
#%%
class NetStrainMat2SegMat(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainMat2SegMat, self).__init__()
        # self.imgDim = 3
        paras = config.get('paras', None)
        n_sectors = paras.get('n_sectors', 18)
        n_frames  = paras.get('n_frames',  25)
        # self.encoder, self.decoder = get3DNet(paras)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            # nn.Conv2d(16, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
            # nn.ReLU(True),
            nn.Conv2d(16, 2, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.Conv2d(2, 2, 1, stride=1, padding=0),  # b, 16, 10, 10
            )
        
    def forward(self, x):
        x = self.net(x)
        return x

class NetStrainImg2TOS(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainImg2TOS, self).__init__()
        # self.imgDim = 3        
        self.paras = config.get('paras', None)
        self.n_input_channels = self.paras.get('n_input_channels', 2)
        self.n_sectors = self.paras.get('n_sectors', 18)
        self.n_frames  = self.paras.get('n_frames',  25)
        self.n_conv_layers = self.paras.get('n_conv_layers',  4)
        self.n_conv_channels = self.paras.get('n_conv_channels',  8)
        self.conv_size = self.paras.get('conv_size',  3)
        H = self.paras.get('H', 64)
        W = self.paras.get('W', 64)
        T = self.paras.get('T', 64)
        if self.n_conv_layers ==1:
            convs = [nn.Conv3d(self.n_input_channels, 1, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        else:
        # self.encoder, self.decoder = get3DNet(paras)
            # convs = [nn.Conv3d(self.n_input_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] + \
            #                 [nn.Conv3d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] * (self.n_conv_layers-2) + \
            #                 [nn.Conv3d(self.n_conv_channels, 1, 1, stride=1, padding=0), nn.ReLU(True)]
                            
            convs = [nn.Conv3d(self.n_input_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
            for midLayerIdx in range(self.n_conv_layers-2):
                convs += [nn.Conv3d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
            convs += [nn.Conv3d(self.n_conv_channels, 1, 1, stride=1, padding=0), nn.ReLU(True)]
                
        linear = [nn.Flatten(), nn.Linear(H*W*T, self.n_sectors)]
        self.layers = nn.ModuleList(convs+linear)
        
        
    def forward(self, x):
        # x = self.net(x)
        for layer in self.layers:
            x = layer(x) 
        return x
    
class NetStrainImg2TOSImg(nn.Module):
    def __init__(self, config = {}):
        super(NetStrainImg2TOSImg, self).__init__()
        # self.imgDim = 3
        self.paras = config.get('paras', None)
        # self.H = paras.get('H', 128)
        # self.W = paras.get('W', 128)
        self.C = self.paras.get('C', 2)
        self.T = self.paras.get('n_frames', 25)
        # n_sectors = paras.get('n_sectors', 18)
        self.dimReduceMethod = self.paras.get('dimReduceMethod',  'linear')
        self.n_conv_layers = self.paras.get('n_conv_layers',  4)
        self.n_conv_channels = self.paras.get('n_conv_channels',  8)
        self.conv_size = self.paras.get('conv_size',  3)
        # convs = [nn.Conv3d(self.C, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] + \
        #         [nn.Conv3d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)] * (self.n_conv_layers-2) + \
        #         [nn.Conv3d(self.n_conv_channels, 1, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
                
        convs = [nn.Conv3d(self.C, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        for midLayerIdx in range(self.n_conv_layers-2):
            convs += [nn.Conv3d(self.n_conv_channels, self.n_conv_channels, self.conv_size, stride=1, padding=1), nn.ReLU(True)]
        convs += [nn.Conv3d(self.n_conv_channels, 1, 1, stride=1, padding=0), nn.ReLU(True)]
        
        
        self.conv_layers = nn.ModuleList(convs)
        self.convlast = nn.Conv2d(self.T, 1, 3, stride = 1, padding = 1)

        
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        # x = torch.squeeze(x)
        # [N, C, T, H, W] = [N, 1, n_frames, H, W] -> [N, n_frames, H, W]
        x = x[:,0,:,:,:]
        x = self.convlast(x)
        # x = torch.reshape(x, (-1, 1, 1, self.H, self.W))
        return x    

    
if __name__ == '__main__':
    import numpy as np
    import torch
    mats = torch.from_numpy(np.random.rand(9, 1, 18, 25).astype(np.float32))
    imgs = torch.from_numpy(np.random.rand(9, 2, 25, 128, 128).astype(np.float32))
    config = {'paras':{'n_sector':18, 'n_frames':25}}
    net = NetStrainImg2TOS(config)
    output = net(imgs).detach().numpy()
    print(output.shape)