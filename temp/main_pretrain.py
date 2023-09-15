# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:53:46 2020

@author: remus
"""
#%% Imports
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from configs.getConfig import getConfigGroup, getDataFilename
from utils.extractCardiacData import getFilenamesGivenPath, extractDataGivenFilenames, \
    mergeDataByPatient,\
    trainTestSplit
from utils.io import saveConfig2Json, createExpFolder, saveStrainDataFig2File
from utils.cardiacUtils import getPatientName
from utils.prepareData import getAllIndices, getIndices
from modules.augmentation import augmentation

#%% 1. Load config
debug = False
showtest = False
saveResults = True
visData = False
loadExtractedData = True
loadExistingConfig = True
resultPath = './results/'
# resultPath = 'D:\\Research\\Cardiac\\Experiment_Results\\'

gpuIdx = 0

def loadConfig(filename):
    with open(filename, 'rb') as f:
        config = pickle.load(f) 
    return config
configName = '../COPIED-2020-Jun-29-useData201-scarFree-strainMat-fixedSpitbyPat/idx-260/idx-260-strainMatSVD-AugShift-LR1.00E-04-CV3-Ch8-config.pkl'
config = loadConfig(configName)
configs = [config]
NConfigs = len(configs)

#%% 2. Prepare Data
dataFilename = '../../../../../Dataset/dataFull-159-2020-06-21.npy'
dataFull = list(np.load(dataFilename, allow_pickle = True))
if configs[0]['data'].get('scarFree', False):
    dataFull = [datum for datum in dataFull if datum['hasScar'] == False]
allDataTypes = list(dataFull[0].keys())

#%% 2. Prepare Data
print('Loading Data...')
if loadExtractedData:
    # dataFilename = '../../Dataset/dataFull-159-2020-06-21.npy'
    # dataPretrainFilename = getDataFilename(configs[0]['data'].get('dataPretrainName', '159'))
    # dataLabeledFilename = getDataFilename(configs[0]['data'].get('dataLabeledName', '159'))
    dataPretrainFilename = '../../../../../Dataset/dataFull-159-2020-06-21.npy'
    dataLabeledFilename = '../../../../../Dataset/dataFull-159-2020-06-21.npy'
    dataPretrainFull = list(np.load(dataPretrainFilename, allow_pickle = True))
    dataLabeledFull = list(np.load(dataLabeledFilename, allow_pickle = True))
    if configs[0]['data'].get('scarFree', False):
        dataPretrainFull = [datum for datum in dataPretrainFull if datum['hasScar'] == False]
        dataLabeledFull = [datum for datum in dataLabeledFull if datum['hasScar'] == False]
    allDataTypes = list(dataPretrainFull[0].keys())
else:    
    allDataTypes = list(set([config['data']['inputType'] for config in configs])) + ['strainMat', 'strainMatSVD'] + \
        list(set([config['data']['outputType'] for config in configs])) + ['TOS'] +\
        ['StrainInfo']+['AnalysisFv']
    dataFilenames, labelFilenames, patient_IDs = getFilenamesGivenPath(loadProcessed=True)
    dataFull, dataFilenamesValid, labelFilenamesValid = extractDataGivenFilenames(dataFilenames[:], labelFilenames = None, dataTypes = allDataTypes, labelInDataFile = True, configs = None)
print('Finished!')

# Create Fake Label for unlabeled (test) data. 
# Here we assume (1) all training data are labeled and 
#                (2) all data have same # of sectors
NSectors = dataLabeledFull[0]['strainMat'].shape[-2]
isFakeTestLabel = False
for datum in dataLabeledFull:
    if 'TOS' not in datum.keys():
        datum['TOS'] = np.zeros((1, NSectors))
        datum['fakeTOS'] = True
        isFakeTestLabel = True



# Get data info
patientPretrainUniqueNames = list(set([getPatientName(datum['dataFilename']) for datum in dataPretrainFull]))
patientLabeledUniqueNames = list(set([getPatientName(datum['dataFilename']) for datum in dataLabeledFull]))
indicesLabeled = getAllIndices(dataLabeledFull, configs[0]['data'].get('Inverse TOS', True))
NPatientsPretrain = len(patientPretrainUniqueNames)
NPatientsLabeled = len(patientLabeledUniqueNames)
if 'strainMat' in allDataTypes:
    NFramesMaxPretrain = np.max([datum['strainMat'].shape[-1] for datum in dataPretrainFull])
    NFramesMaxLabeled = np.max([datum['strainMat'].shape[-1] for datum in dataLabeledFull])
    NFramesMax = max(NFramesMaxPretrain, NFramesMaxLabeled)
elif 'dispField' in allDataTypes:
    NFramesMax = np.max([datum['dispField'].shape[2] for datum in dataFull])
NFramesMax = int(NFramesMax)

# Add Patinent Number
for datum in dataPretrainFull:
    datum['patientNo'] = patientPretrainUniqueNames.index(getPatientName(datum['dataFilename']))
for datum in dataLabeledFull:
    datum['patientNo'] = patientLabeledUniqueNames.index(getPatientName(datum['dataFilename']))    

#%% 3. Create Experiment Results Folder
if saveResults:
    expgPath, expgName = createExpFolder(resultPath, configName, create_subFolder=False, addDate = True)    
    
#%% 4. Run Experiments
for expIdx, config in enumerate(configs):
    #%% 4.1 Extrac and Save config
    print(f'Start config {expIdx} / {len(configs)-1}')
    print(config)
    expFoldername = config['name'] if len(config['name'])>0 else f'idx-{expIdx + startIdx}'
    expPath, expName = createExpFolder(expgPath, expFoldername, create_subFolder=False, addDate = False) 
    
    saveConfig2Json(config, expPath + 'config.json')
    # https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
    with open(expPath + config['name'] + '-config.pkl', 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
        
    #%% 4.2 Extract Used Data
    ifMergeDataByPatient = config['data'].get('mergeAllSlices', False)
    if ifMergeDataByPatient:
        # strainMat: [1, 1, NSectos, NFrames] -> [1, NSlices, NSectors, NFrames]
        dataPretrainFullUsed = mergeDataByPatient(dataPretrainFull, patientPretrainUniqueNames, reorderByLoc = True)
        dataLabeledFullUsed = mergeDataByPatient(dataLabeledFull, patientLabeledUniqueNames, reorderByLoc = True)
        NSlicesMax = np.max([datum['strainMat'].shape[1] for datum in dataPretrainFullUsed])
    else:
        dataPretrainFullUsed = dataPretrainFull
        dataLabeledFullUsed = dataLabeledFull
    
    train_indices, test_indices = getIndices(indicesLabeled, ifMergeDataByPatient, config['data']['train_test_split'])
    dataTr = [dataLabeledFullUsed[idx] for idx in train_indices]
    dataTe = [dataLabeledFullUsed[idx] for idx in test_indices]
    for datum in dataTr: datum['isTraining'] = True
    for datum in dataTe: datum['isTraining'] = False
        
    #%% 4.3 Pre-processing and Augmentation
    # Unify Shape
    from utils.cardiacUtils import changeNFrames, changeNSlices
    # Pretrain data: only process input
    for dataType in set([config['data']['inputType'],'strainMatSVD']):
        for datum in dataPretrainFullUsed: changeNFrames(datum, NFramesMax, dataType)
        if config['data'].get('mergeAllSlices', False):
            for datum in dataPretrainFullUsed: changeNSlices(datum, NSlicesMax, dataType, config['data'].get('mergeAllSlices', False))
            
    # Training data: both input and output data
    for dataType in set([config['data']['inputType'],config['data']['outputType'], 'strainMatSVD', 'TOS']):
        for datum in dataTr + dataTe: changeNFrames(datum, NFramesMax, dataType)
        if config['data'].get('mergeAllSlices', False):
            for datum in dataTr + dataTe: changeNSlices(datum, NSlicesMax, dataType, config['data'].get('mergeAllSlices', False))
    
    # Truncate zeros to speed up training
    yMin, yMax = 30, 98
    xMin, xMax = 30, 98
    if config['data']['inputType'] in ['dispField']:
        config['data']['truncate']={'yMin':xMin,'yMax':xMax,'xMin':xMin,'xMax':xMax}
        for datum in dataPretrainFullUsed + dataTr + dataTe:
            datum[config['data']['inputType']] = datum[config['data']['inputType']][:,:,:,yMin:yMax,xMin:xMax]
            datum['spatialMask'] = datum['spatialMask'][:,:,:,yMin:yMax,xMin:xMax]        
            datum['analysisFv'].vertices -= np.array([xMin,yMin])
    if config['data']['outputType'] in ['TOSImage']:
        for datum in dataTr + dataTe:
            datum[config['data']['outputType']] = datum[config['data']['outputType']][:,:,yMin:yMax,xMin:xMax]
    
    # Augmentation
    if len(config['data']['augmentation']) > 0:
        augConfig = config['data']['augmentation']
        augConfig['inputType'] = config['data']['inputType']
        augConfig['outputType'] = config['data']['outputType']
        augmentation(dataPretrainFullUsed, augConfig)
        augmentation(dataTr, augConfig)
        # inputDataTr, labelDataTr = augmentation(inputDataTrRaw, labelDataTrRaw, augConfig)
        
    

    #%% 4.4 Prepare input data
    # Make 3D data: 
    # strainMat: [N, NSlices, NSectors, NFrames] -> [N, 1, NSlices, NSectors, NFrames]
    # TOS      : [N, NSlice, 1, NSectors] -> [N, 1, NSlice, 1, NSectors]
    if config['data']['inputType'] in ['strainMat', 'strainMatSVD']:
        if ifMergeDataByPatient:
            for datum in dataPretrainFullUsed + dataTr + dataTe:
                datum[config['data']['inputType']] = datum[config['data']['inputType']][:,None,:,:,:]
            
    if config['data']['outputType'] in ['TOS']:
        if ifMergeDataByPatient:
            for datum in dataTr + dataTe:
                datum[config['data']['outputType']] = datum[config['data']['outputType']][:,None,:,:,:]
    
    inputPretrain = np.concatenate([datum[config['data']['inputType']] for datum in dataPretrainFullUsed], axis=0)
    labelPretrain = np.array([datum['patientNo'] for datum in dataPretrainFullUsed])    
    
    inputDataTr = np.concatenate([datum[config['data']['inputType']] for datum in dataTr], axis=0)
    inputDataTe = np.concatenate([datum[config['data']['inputType']] for datum in dataTe], axis=0)
    labelDataTr = np.concatenate([datum[config['data']['outputType']] for datum in dataTr], axis=0)
    labelDataTe = np.concatenate([datum[config['data']['outputType']] for datum in dataTe], axis=0)
    
    device = torch.device(f"cuda:{gpuIdx}" if torch.cuda.is_available() else "cpu")
    from modules.DataSet import DataSet2D, DataSetContrastive
    datasetPretrain = DataSetContrastive(inputPretrain, labelPretrain, device)
    datasetTr = DataSet2D(imgs = inputDataTr, labels = labelDataTr, transform=None, device = device)
    datasetTe = None if isFakeTestLabel else DataSet2D(imgs = inputDataTe, labels = labelDataTe, transform=None, device = device)            

    #%% 4.5 Get Networks
    from modules.netModule import NetModule
    from modules.strainNets import getNetwork
    from modules.compNets import NetComp
    net_config = config['net']
    net_pretrain_config = config['net_pretrain']
    for netConf in [net_config, net_pretrain_config]:
        netConf['paras']['n_frames'] = NFramesMax
        netConf['inputType'] = config['data']['inputType']
        netConf['inputShape'] = inputDataTr.shape
        netConf['outputType'] = config['data']['outputType']    
        netConf['n_slices'] = inputDataTr.shape[2] if ifMergeDataByPatient else None
        netConf['mergeAllSlices'] = ifMergeDataByPatient
    netPretrain = NetModule(getNetwork(net_pretrain_config), config['loss_pretrain'], device)
    net = NetModule(NetComp(netPretrained = netPretrain.net, netFinetune = getNetwork(net_config)), config['loss'], device)
    
    #%% 4.6 Pre-Training
    if loadExistingConfig:
        netFile = './temp/idx-1-strainMat-AugNone-LR1.00E-04-CV5-Ch4-net.pth'
        net.load(netFile, map_location={'cuda:1': f'cuda:{gpuIdx}','cuda:2': f'cuda:{gpuIdx}','cuda:3': f'cuda:{gpuIdx}'})
    else:
        if debug:
            config['training']['learning_rate'] = 1e-4
            config['training']['epochs_num'] = 50
            config['training']['batch_size'] = 10
        if config['training']['batch_size'] == 'half':
            config['training']['batch_size'] = max(len(dataTr)//2, 10)
        if config['pretrain']['batch_size'] == 'half':
            config['pretrain']['batch_size'] = max(len(dataTr)//2, 10)
        #loss_history, validLoss_history, past_time = net.train(training_dataset=training_dataset, training_config = config['training'], valid_dataset = test_dataset, expPath = None)
        print('Pre-training...')
        pretrain_loss_history, _, pretrain_past_time = netPretrain.train(datasetPretrain, training_config = config['pretrain'], valid_dataset = None, expPath = None)
        print('Fine-tuning...')
        loss_history, validLoss_history, past_time = net.train(training_dataset=datasetTr, training_config = config['training'], valid_dataset = datasetTe, expPath = None, finetune = True)
    
    #%% 6. Validate Model
    # Get Prediction
    for datum in dataTr + dataTe:
        # datum['dataFilename'][datum['isOriSlice'] == False] = 'Interpreted'
        # for sidx in range(NSlicesMax): datum['dataFilename']
        datum['predRaw'], datum['loss'] = net.pred(datum[config['data']['inputType']], labels = datum[config['data']['outputType']], avg = True)
    # predTrRaw, lossTr = net.pred(inputDataTr, labels = labelDataTr, avg = True)
    # predTeRaw, lossTe = net.pred(inputDataTe, labels = labelDataTe, avg = True)
    
    # Transform Prediction into TOS if necessary
    for datum in dataTr + dataTe:
        if config['data']['outputType'] == 'TOSImage':
            pass
            # from utils.cardiacUtils import mat2recvfv, rectfv2TOS
            # analysisFvRaw = [datum['AnalysisFv'] for datum in dataFullUsed]
            
            # analysisFvTr = [analysisFvRaw[idx] for idx in train_indices]
            # analysisFvTe = [analysisFvRaw[idx] for idx in test_indices]
            # predTrFvVals = [mat2recvfv(analysisFvTr[idx].faces, analysisFvTr[idx].vertices, np.squeeze(predTrRaw[idx,:])) for idx in range(predTrRaw.shape[0])]
            # predTeFvVals = [mat2recvfv(analysisFvTe[idx].faces, analysisFvTe[idx].vertices, np.squeeze(predTeRaw[idx,:])) for idx in range(predTeRaw.shape[0])]
            # predTr = [rectfv2TOS(predTrFvVals[idx], analysisFvTr[idx].sectorid, analysisFvTr[idx].layerid).reshape((1,-1)) for idx in range(predTrRaw.shape[0])]
            # predTr = np.concatenate(predTr, axis = 0)
            # predTe = [rectfv2TOS(predTeFvVals[idx], analysisFvTe[idx].sectorid, analysisFvTe[idx].layerid).reshape((1,-1)) for idx in range(predTeRaw.shape[0])]
            # predTe = np.concatenate(predTe, axis = 0)
        elif config['data']['inputType'] == 'strainCurve' and config['data']['outputType']=='TOS':
            # data shape [N*NS, NF] -> [M, 1, NS, NF]
            # pred shape [N*NS] -> [N, NS]
            pass
            # training_dataset.data = training_dataset.data.reshape((-1,1,NS,NF))
            # training_dataset.labels = training_dataset.labels.reshape((-1, NS))
            # test_dataset.data = test_dataset.data.reshape((-1,1,NS,NF))
            # test_dataset.labels = test_dataset.labels.reshape((-1, NS))
            # predTr = predTrRaw.reshape((-1,NS))
            # predTe = predTeRaw.reshape((-1,NS))
        else:
            datum['pred'] = datum['predRaw']
            # predTr = predTrRaw
            # predTe = predTeRaw
    
    # Reshape TOS. Should be [N, NSectors] if not merge slices else [N, NSlices, NSectors]
    # predTr, predTe = np.squeeze(predTr), np.squeeze(predTe)
    
    # Visualize
    if showtest:
        from IPython import get_ipython
        get_ipython().magic('matplotlib auto')
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
        
        # Visulize training data prediction and GT
        # if np.ndim(predTr) ==3:
        #     # if [N, NSlices, NSectors]
        #     pass
        # else:
        #     pass
        
        #
        # pp = 2
        # for ii in range(9):
        #     visTOsS([labelDataTr[pp,ii,0,:], predTr[pp,ii,0,:]], inputDataTr[pp,0,ii,:,:], ['GT', 'Esti'])
        # #
        # if not isFakeTestLabel:
        #     ii = 1
        #     visTOsS([TOSDataTrRaw[ii,:], predTr[ii,:]], np.squeeze(EccMatDataTrRaw[ii]), ['GT','Esti'], 'training')
        #     for ii in range(min(len(test_dataset), 5)):
        #         visTOsS([TOSDataTeRaw[ii,:], predTe[ii,:]], np.squeeze(EccMatDataTeRaw[ii]), ['GT','Esti'], f'test {ii+1}')
        
    # Save Figures to file
    if debug:
        savePath = './temp/imgs/'
    else:
        savePath = expPath
    if saveResults:
        from utils.cardiacUtils import getSliceName
        
        from utils.io import saveLossInfo
        # saveLossInfo(predTe, labelDataTe, savePath)
        
        if ifMergeDataByPatient:
            # if data are merged by patient, save only original slices 
            for idx, datum in enumerate(dataTr + dataTe):
                if not datum.get('isAugmented', False):
                    # Skip augmented data
                    sliceIDs = [None] * NSlicesMax
                    orgSIdx = 0
                    for sidx in range(NSlicesMax):
                        if datum['isOriSlice'][sidx] == True:
                            sliceIDs[sidx] = datum['dataFilename'][orgSIdx]
                            orgSIdx += 1
                        else:
                            sliceIDs[sidx] = 'Interperted'
                    MSEPerSlice = np.squeeze(np.linalg.norm(datum['pred'] - datum['TOS'], axis=-1))
                    title = '↑ ' + config['name'] + '\n' + configName
                    trainTestStr = f'train {idx}' if idx < len(dataTr) else f'test {idx - len(dataTr)}'
                    subtitles = [f'{trainTestStr} slice {sidx}' \
                              + '\n'+ getSliceName(sliceIDs[sidx]) \
                              + f'\nsliceLoc {datum["sliceLocsInterped"][sidx]:.2f}' \
                              + f'\nMSE = {MSEPerSlice[sidx]:.2f}'
                              for sidx in range(datum['pred'].shape[2]) if datum['isOriSlice'][sidx] == True]
                        
                    saveFilename = savePath + (f'train_{idx}.png' if idx < len(dataTr) else f'test {idx - len(dataTr)}')
                    print('Saving' + saveFilename)
                    # print(np.squeeze(datum['TOS']))
                    saveStrainDataFig2File(datum['strainMatSVD'][0,datum['isOriSlice']], 
                        [np.squeeze(datum['TOS'])[datum['isOriSlice']], np.squeeze(datum['pred'])[datum['isOriSlice']]],
                        saveFilename, 
                        legends = ['GT', 'Pred'], 
                        title = title, subtitles = subtitles,
                        inverseTOS = not config['data'].get('Inverse TOS', False))
        else:
            # Else, save all slices. Organized by patient
            for patientIdx, patientName in enumerate(patientLabeledUniqueNames):
                #dataOfPatient, dataIdxs = zip(*[(datum, datumIdx) for (datumIdx, datum) in enumerate(dataTr + dataTe) if patientName in datum.get('dataFilename', 'NONE') and not datum.get('isAugmented', False)])
                dataOfPatient, dataIdxs = zip(*[(datum, datumIdx) for (datumIdx, datum) in enumerate(dataTr + dataTe) if patientName in datum.get('dataFilename', 'NONE')])
                sliceLocs = [datum['SequenceInfo'] for datum in dataOfPatient]
                dataOfPatientSorted = [dataOfPatient[idx] for idx in np.argsort(sliceLocs)]
                dataIdxsSorted = [dataIdxs[idx] for idx in np.argsort(sliceLocs)]
                sliceLocsSorted = [sliceLocs[idx] for idx in np.argsort(sliceLocs)]
                
                strainMats = np.squeeze(np.concatenate([datum['strainMatSVD'] for datum in dataOfPatientSorted]))
                TOSs = np.concatenate([datum['TOS'] for datum in dataOfPatientSorted])
                preds = np.concatenate([datum['pred'] for datum in dataOfPatientSorted])
                MSEs = np.linalg.norm(TOSs - preds, axis=-1)
                title = '↑ ' + config['name'] + '\n' + configName
                subtitles = [(f'train {dataIdxs[sidx]}' if dataIdxsSorted[sidx]<len(dataTr) else f'test {dataIdxs[sidx] - len(dataTr)}') \
                             + f'\n {getSliceName(dataOfPatientSorted[sidx]["dataFilename"])}' \
                             + f'\n SliceLoc = {sliceLocsSorted[sidx]:.2f}'
                             + f'\n MSE={MSEs[sidx]:.2f}'
                             for sidx in range(len(dataOfPatientSorted))]
                saveFilename = savePath + f'result_{patientIdx}'
                saveStrainDataFig2File(strainMats, 
                        [TOSs, preds],
                        saveFilename, 
                        legends = ['GT', 'Pred'], 
                        title = title, subtitles = subtitles,
                        inverseTOS = not config['data'].get('Inverse TOS', False))
                
    config['performance'] = {
        'lossTrBatch': loss_history[-1],
        'lossTr': sum([datum['loss'] for datum in dataTr]),
        'lossTe': sum([datum['loss'] for datum in dataTe])
        }
    
    net.saveLossHistories([loss_history, validLoss_history], savePath + 'losses.png', 
                          report_epochs_num=config['training']['report_per_epochs'], 
                          legends=['training loss', 'test loss'])
    
    net.save(savePath + config['name'] + '-net.pth')
    
    saveConfig2Json(config, savePath + 'config.json')
    # https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
    with open(savePath + config['name'] + '-config.pkl', 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

if saveResults:
    with open(expgPath + 'configs.pkl', 'wb') as f:
        pickle.dump(configs, f, pickle.HIGHEST_PROTOCOL)
