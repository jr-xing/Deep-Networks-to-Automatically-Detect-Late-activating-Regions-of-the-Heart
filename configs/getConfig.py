# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:28:31 2019

@author: Jerry Xing
"""
import copy
import numpy as np
def translateNNIArgs2Config(config, args):
    # for aug in [{},{'shiftY':list(range(-5,6))},{'shiftY':list(range(-9,9))}]:
    for key in args.keys():
        if key == 'aug':
            if args[key] == 0:
                config['data']['augmentation'] = {}
            elif args[key] == 11:
                config['data']['augmentation'] = {'shiftY':list(range(-5,6))}
            elif args[key] == 18:
                config['data']['augmentation'] = {'shiftY':list(range(-9,9))}
        elif key in ['inputType', 'outputType', 'scarFree', 'train_test_split', 'paddingMethod']:
            if key == 'scarFree':
                config['data'][key] = bool(args[key])            
            else:
                config['data'][key] = args[key]
        elif key in ['n_conv_layers', 'n_conv_channels']:
            config['net']['paras'][key] = args[key]
        elif key in ['learning_rate']:
            config['training'][key] = args[key]
            
    return config

def getDataFilename(dataName):
    if dataName == '159':
        dataFilename = '../../Dataset/dataFull-159-2020-06-21.npy'
    elif dataName == '201':
        dataFilename = '../../Dataset/dataFull-201-2020-06-27.npy'
    return dataFilename

def getConfig(name):
    if name == 'default':
        config = {
            'name': '',
            'comment': '',
            'data':{
                'inputType': 'strainMat',
                'outputType': 'TOS',
                'outlierThres': 0,
                'train_test_split': 'random',
                'augmentation':{}
                },
            'net':
                {
                    'type': 'StrainMat2TOS',                        
                    'paras': {
                        'n_sector':18, 
                        'n_frames':25
                            }                        
                 },
            'training':{
                    'epochs_num': 2000,
                    'batch_size': 30,
                    'learning_rate':1e-5,
                    'report_per_epochs':20,
                    'training_check': False,
                    'valid_check': True,
                    'save_trained_model': True},
            'loss':{
                    'name': 'strainMat1D',
    #                'name': 'TV',
                    'para': None
                    }
        }
    elif name == 'pretrain_default':
        config = {
            'name': '',
            'comment': '',
            'data':{
                'inputType': 'strainMat',
                'outputType': 'TOS',
                'outlierThres': 0,
                'train_test_split': 'random',
                'augmentation':{}
                },
            'net':
                {
                    'type': 'simpleFCN',                        
                    'paras': {
                        'n_sector':18, 
                        'n_frames':25
                            }                        
                 },
            'net_pretrain':{
                'type': 'Siamese',
                'paras': {
                        'n_sector':18, 
                        'n_frames':25
                            }                        
                },                
            'training':{
                    'epochs_num': 2000,
                    'batch_size': 'half',
                    'learning_rate':1e-4,
                    'report_per_epochs':20,
                    'training_check': False,
                    'valid_check': True,
                    'save_trained_model': True},
            'pretrain':{
                    'contrastive': True,
                    'epochs_num': 2000,
                    'batch_size': 'half',
                    'learning_rate':1e-4,
                    'report_per_epochs':20,
                    'training_check': False,
                    'valid_check': False,
                    'save_trained_model': False
                },
            'loss':{
                    'name': 'strainMat1D',
    #                'name': 'TV',
                    'para': None
                    },
            'loss_pretrain':{
                'name': 'contrastive',
                'para': None
                }
        }
    elif name == 'debug':
        config = getConfig('default')
        config['data']['augmentation'] = None
        config['net']['paras']['name'] = 'debug'
        config['training']['epochs_num'] = 50
        config['training']['report_per_epochs'] = 5

    
    return config

def getConfigGroup(name):
    configs = []
    if name == 'debug':
        configName = 'debug'
        config = getConfig(configName)
        config['idxInGroup'] = 1
        configs.append(config)

        config = getConfig(configName)
        config['idxInGroup'] = 2
        config['training']['batch_size'] = 100
        configs.append(config)
        
    elif name == 'Constrastive Pre-train - Test':
        configDefault = getConfig('pretrain_default')        
        configs = [configDefault]
        
    elif name == 'useData159-scarFree-orgDataByPatient-strainMat-interp-lrelu' \
        or name == 'useData159-scarFree-orgDataByPatient-strainMat-interp-lreluCorr':
        
        configDefault = getConfig('default')        
        configIdx = 0
        configDefault['data']['dataName'] = '159'
        configDefault['data']['paddingMethod'] = 'zero'
        configDefault['data']['mergeAllSlices'] = 'interp'
        configDefault['data']['scarFree'] = True
        configDefault['data']['train_test_split'] = 'fixed'
        configDefault['training']['batch_size'] = 'half'
        
        for aug in [{},{'shiftY':list(range(-5,6))},{'shiftY':list(range(-9,9))}]:
            for n_conv3d_layers in [6,4,2]:
                for n_conv2d_layers in [6,4,2]:
                    for n_conv3d_channels in [8, 16, 24]:
                        for n_conv2d_channels in [8, 16, 24]:
                            for lr in [1e-3, 1e-4, 1e-5]:
                                config = copy.deepcopy(configDefault)
                                augstr = 'AugNone' if aug == {} else 'AugShift'
                                config['name'] = f'idx-{configIdx}-LR{lr:.2E}-{augstr}-CV2D-{n_conv2d_layers}-Ch3D{n_conv2d_channels}-CV3D{n_conv3d_layers}-Ch3D{n_conv2d_channels}'
                                config['data']['augmentation'] = aug
                                config['net']['paras']['n_conv2d_layers'] = n_conv2d_layers
                                config['net']['paras']['n_conv3d_layers'] = n_conv3d_layers
                                config['net']['paras']['n_conv2d_channels'] = n_conv2d_channels
                                config['net']['paras']['n_conv3d_channels'] = n_conv3d_channels
                                config['training']['learning_rate'] = lr
                                
                                configs.append(config)
                                configIdx += 1
                                
    elif name == 'useData159-scarFree-strainMat-fixedSpitbyPat':
        configDefault = getConfig('default')
        configDefault['data']['dataName'] = '159'
        # configDefault['data']['filename'] = ''
        configDefault['data']['train_test_split'] = 'fixedPatient' 
        configDefault['data']['scarFree'] = True        
        configDefault['data']['paddingMethod'] = 'zero'
        configIdx = 0
        for inputType in ['strainMat']:
            for n_conv_layers in [5,4,3,2]:
                for n_conv_channels in [4,8, 16, 24]:
                        for aug in [{},{'shiftY':list(range(-5,6))},{'shiftY':list(range(-9,9))}]:
                                for lr in [1e-3, 1e-4, 1e-5]:
                                    config = copy.deepcopy(configDefault)
                                    augstr = 'AugNone' if aug == {} else 'AugShift'
                                    config['name'] = f'idx-{configIdx}-{inputType}-{augstr}-LR{lr:.2E}-CV{n_conv_layers}-Ch{n_conv_channels}'
                                    config['data']['inputType'] = inputType
                                    config['data']['augmentation'] = aug
                                    if inputType in ['strainMat', 'strainMatSVD', 'dispFieldJacoMat']:
                                        if 'shiftX' in config['data']['augmentation'].keys():
                                            config['data']['augmentation']['shiftX'] = [0]
                                    config['net']['paras']['n_conv_layers'] = n_conv_layers
                                    config['net']['paras']['n_conv_channels'] = n_conv_channels
                                    config['training']['learning_rate'] = lr
                                    configs.append(config)
                                    configIdx += 1
                                    
    elif name == 'useData159-scarFree-orgDataByPatient-strainMat-interp-lrelu':
        configDefault = getConfig('default')
        configIdx = 0
        configDefault['data']['paddingMethod'] = 'zero'
        configDefault['data']['mergeAllSlices'] = 'interp'
        configDefault['data']['scarFree'] = True
        configDefault['data']['train_test_split'] = 'fixed'
        configDefault['training']['batch_size'] = 10
        
        for aug in [{},{'shiftY':list(range(-5,6))},{'shiftY':list(range(-9,9))}]:
            for n_conv3d_layers in [2,4,6]:
                for n_conv2d_layers in [2,4,6]:
                    for n_conv3d_channels in [8, 16, 24]:
                        for n_conv2d_channels in [8, 16, 24]:
                            for lr in [1e-3, 1e-4, 1e-5]:
                                config = copy.deepcopy(configDefault)
                                augstr = 'AugNone' if aug == {} else 'AugShift'
                                config['name'] = f'idx-{configIdx}-LR{lr:.2E}-{augstr}-CV2D-{n_conv2d_layers}-Ch3D{n_conv2d_channels}-CV3D{n_conv3d_layers}-Ch3D{n_conv2d_channels}'
                                config['data']['augmentation'] = aug
                                config['net']['paras']['n_conv2d_layers'] = n_conv2d_layers
                                config['net']['paras']['n_conv3d_layers'] = n_conv3d_layers
                                config['net']['paras']['n_conv2d_channels'] = n_conv2d_channels
                                config['net']['paras']['n_conv3d_channels'] = n_conv3d_channels
                                config['training']['learning_rate'] = lr
                                
                                configs.append(config)
                                configIdx += 1
                            
    elif name == 'useData161-scarFree-strainMat-fixedSpitbyPat':
        configDefault = getConfig('default')
        # configDefault['data']['filename'] = ''
        configDefault['data']['train_test_split'] = 'fixedPatient' 
        configDefault['data']['scarFree'] = True
        configDefault['data']['paddingMethod'] = 'zero'
        configIdx = 0
        for inputType in ['strainMat', 'strainMatSVD']:
            for n_conv_layers in [2,3,4,5]:
                for n_conv_channels in [4,8, 16, 24]:
                        for aug in [{},{'shiftY':list(range(-5,6))},{'shiftY':list(range(-9,9))}]:
                                for lr in [1e-4, 5e-5, 1e-5, 5e-6]:
                                    config = copy.deepcopy(configDefault)
                                    augstr = 'AugNone' if aug == {} else 'AugShift'
                                    config['name'] = f'idx-{configIdx}-{inputType}-{augstr}-LR{lr:.2E}-CV{n_conv_layers}-Ch{n_conv_channels}'
                                    config['data']['inputType'] = inputType
                                    config['data']['augmentation'] = aug
                                    if inputType in ['strainMat', 'strainMatSVD', 'dispFieldJacoMat']:
                                        if 'shiftX' in config['data']['augmentation'].keys():
                                            config['data']['augmentation']['shiftX'] = [0]
                                    config['net']['paras']['n_conv_layers'] = n_conv_layers
                                    config['net']['paras']['n_conv_channels'] = n_conv_channels
                                    config['training']['learning_rate'] = lr
                                    configs.append(config)
                                    configIdx += 1
    elif name == 'useData157-orgDataByPatient-strainMat':
        configDefault = getConfig('default')
        configIdx = 0
        configDefault['data']['paddingMethod'] = 'zero'
        configDefault['data']['mergeAllSlices'] = True
        configDefault['training']['batch_size'] = 10
        for aug in [{},{'shiftY':list(range(-5,6))},{'shiftY':list(range(-9,9))}]:
            for n_conv3d_layers in [2,4,6]:
                for n_conv2d_layers in [2,4,6]:
                    for n_conv3d_channels in [8, 16, 24]:
                        for n_conv2d_channels in [8, 16, 24]:
                            for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
                                config = copy.deepcopy(configDefault)
                                augstr = 'AugNone' if aug == {} else 'AugShift'                                
                                config['name'] = f'idx-{configIdx}-LR{lr:.2E}-{augstr}-CV2D-{n_conv2d_layers}-Ch3D{n_conv2d_channels}-CV3D{n_conv3d_layers}-Ch3D{n_conv2d_channels}'
                                config['data']['augmentation'] = aug
                                config['net']['paras']['n_conv2d_layers'] = n_conv2d_layers
                                config['net']['paras']['n_conv3d_layers'] = n_conv3d_layers
                                config['net']['paras']['n_conv2d_channels'] = n_conv2d_channels
                                config['net']['paras']['n_conv3d_channels'] = n_conv3d_channels
                                config['training']['learning_rate'] = lr
                                configs.append(config)
                                configIdx += 1
        
    elif name == 'useData157-dispFieldFvTuning':        
        configDefault = getConfig('default')
        configIdx = 0
        configDefault['data']['inputType'] = 'dispFieldFv'
        configDefault['data']['paddingMethod'] = 'zero'
        for n_conv_layers in [3, 4, 5]:
            for n_conv_channels in [8, 16, 24]:
                for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
                    config = copy.deepcopy(configDefault)
                    config['name'] = f'idx-{configIdx}-LR{lr:.2E}-CV{n_conv_layers}-Ch{n_conv_channels}'                            
                    config['net']['paras']['n_conv_layers'] = n_conv_layers
                    config['net']['paras']['n_conv_channels'] = n_conv_channels
                    config['training']['learning_rate'] = lr
                    configs.append(config)
                    configIdx += 1

    elif name == 'useData157-dispFieldTuning':
        configDefault = getConfig('default')
        configIdx = 0
        configDefault['data']['inputType'] = 'dispField'
        configDefault['data']['paddingMethod'] = 'zero'
        for n_conv_layers in [3,4,5]:
            for n_conv_channels in [8, 16, 24]:
                for lr in [1e-4, 1e-5, 1e-6]:
                    config = copy.deepcopy(configDefault)
                    config['name'] = f'idx-{configIdx}-LR{lr:.2E}-CV{n_conv_layers}-Ch{n_conv_channels}'                            
                    config['net']['paras']['n_conv_layers'] = n_conv_layers
                    config['net']['paras']['n_conv_channels'] = n_conv_channels
                    config['training']['learning_rate'] = lr
                    configs.append(config)
                    configIdx += 1
                                    
                                    
    elif name == 'useData157-strainMatTuning-Padding':
        configDefault = getConfig('default')
        configIdx = 0
        for inputType in ['strainMat', 'strainMatSVD']:
            for n_conv_layers in [3,4,5]:
                for n_conv_channels in [8, 16, 24]:
                    for paddingMethod in ['zero', 'circular']:
                        for aug in [{},{'shiftY':list(range(-5,6))},{'shiftY':list(range(-9,9))}]:
                            for outlierThres in [0, 30]:
                                for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6]:
                                    config = copy.deepcopy(configDefault)
                                    augstr = 'AugNone' if aug == {} else 'AugShift'
                                    config['name'] = f'idx-{configIdx}-{inputType}-{augstr}-OutlierThres{outlierThres}-LR{lr:.2E}-CV{n_conv_layers}-Ch{n_conv_channels}'
                                    config['data']['inputType'] = inputType
                                    config['data']['augmentation'] = aug
                                    if inputType in ['strainMat', 'strainMatSVD', 'dispFieldJacoMat']:
                                        if 'shiftX' in config['data']['augmentation'].keys():
                                            config['data']['augmentation']['shiftX'] = [0]
                                    config['data']['outlierThres'] = outlierThres
                                    config['data']['paddingMethod'] = paddingMethod
                                    config['net']['paras']['n_conv_layers'] = n_conv_layers
                                    config['net']['paras']['n_conv_channels'] = n_conv_channels
                                    config['training']['learning_rate'] = lr
                                    configs.append(config)
                                    configIdx += 1
                        
    elif name == 'useData157-TOSImage':
        configDefault = getConfig('default')
        configDefault['data']['inputType'] = 'dispField'
        configDefault['data']['outputType'] = 'TOSImage'
        configIdx = 0
        # # of network conv layers, lr, padding method, 
        for n_conv_layers in [3,4,5]:
            for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
                for paddingMethod in ['zero', 'circular']:
                    config = copy.deepcopy(configDefault)                    
                    config['name'] = f'idx-{configIdx}-CV{n_conv_layers}-LR{lr:.2E}-PD{paddingMethod}'
                    config['data']['inputType'] = 'dispField'
                    config['data']['outType'] = 'TOSImage'
                    config['training']['learning_rate'] = lr
                    config['data']['paddingMethod'] = paddingMethod                                        
                    configs.append(config)
                    configIdx += 1
        
    elif name == 'useData148-first':
        # data type, Learning rate, augmentation, outlier
        configDefault = getConfig('default')
        configIdx = 0
        for inputType in ['strainMat', 'strainMatSVD', 'dispFieldJacoMat', 'dispField', 'dispFieldJacoImg', 'strainImg']:
            for aug in [{},{'shiftX':list(range(-5,6)), 'shiftY':list(range(-5,6))}]:
                for outlierThres in [0, 30]:
                    for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
                        config = copy.deepcopy(configDefault)
                        augstr = 'AugNone' if aug == {} else 'AugShift'
                        config['name'] = f'idx-{configIdx}-{inputType}-{augstr}-OutlierThres{outlierThres}-LR{lr:.2E}'
                        config['data']['inputType'] = inputType
                        config['data']['augmentation'] = aug
                        if inputType in ['strainMat', 'strainMatSVD', 'dispFieldJacoMat']:
                            if 'shiftX' in config['data']['augmentation'].keys():
                                config['data']['augmentation']['shiftX'] = [0]
                        config['data']['outlierThres'] = outlierThres
                        config['training']['learning_rate'] = lr
                        configs.append(config)
                        configIdx += 1
        
        

    return configs