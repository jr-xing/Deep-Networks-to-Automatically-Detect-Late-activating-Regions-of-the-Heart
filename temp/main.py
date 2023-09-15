# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:28:00 2020

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
from modules.augmentation import augmentation
import copy

#%% 1. Load config
debug = False
showtest = False
saveResults = True
visData = False
loadExtractedData = True
loadExistingConfig = True
resultPath = '../results/'
# resultPath = 'D:\\Research\\Cardiac\\Experiment_Results\\'
gpuIdx = 0

if loadExistingConfig:
    def loadConfig(filename):
        with open(filename, 'rb') as f:
            config = pickle.load(f) 
        return config
    # configName = '../COPIED-2020-Jun-29-useData201-scarFree-strainMat-fixedSpitbyPat/idx-260/idx-260-strainMatSVD-AugShift-LR1.00E-04-CV3-Ch8-config.pkl'
    configName = './temp/idx-260-strainMatSVD-AugShift-LR1.00E-04-CV3-Ch8-config-retrain.pkl'
    config = loadConfig(configName)
    configs = [config]
else:
    # configName = 'useData159-scarFree-orgDataByPatient-strainMat-interp-lrelu'
    configName = 'Constrastive Pre-train - Test'
    configs = getConfigGroup(configName)

# configs = [config for config in configs if config['net']['paras']['n_conv_layers']>3]
NConfigs = len(configs)
# startIdx = 0
# endIdx = NConfigs//2
# gpuIdx = 0

startIdx = NConfigs//2
endIdx = NConfigs
gpuIdx = 0

configs = configs[startIdx:endIdx]

#%% 2. Prepare Data
print('Loading Data...')
if loadExtractedData:
    # dataFilename = '../../Dataset/dataFull-159-2020-06-21.npy'
    dataFilename = '../../../../../Dataset/dataFull-159-2020-06-21.npy'
    # dataFilename = '../../../../../Dataset/dataFull-201-2020-06-27.npy'
    # dataFilename = getDataFilename(configs[0]['data'].get('dataName', '159'))
    dataFull = list(np.load(dataFilename, allow_pickle = True))
    if configs[0]['data'].get('scarFree', False):
        dataFull = [datum for datum in dataFull if datum['hasScar'] == False]
        
    allDataTypes = list(dataFull[0].keys())
    dataFilenamesValid = list(datum['dataFilename'] for datum in dataFull)
else:    
    allDataTypes = list(set([config['data']['inputType'] for config in configs])) + ['strainMat', 'strainMatSVD'] + \
        list(set([config['data']['outputType'] for config in configs])) + ['TOS'] +\
        ['StrainInfo']+['AnalysisFv']
    dataFilenames, labelFilenames, patient_IDs = getFilenamesGivenPath(loadProcessed=True)
    dataFull, dataFilenamesValid, labelFilenamesValid = extractDataGivenFilenames(dataFilenames[:], labelFilenames = None, dataTypes = allDataTypes, labelInDataFile = True, configs = None)
print('Finished!')


    

from utils.prepareData import getIndices
# indices = getIndices(dataFull, configs[0])
# Create Fake Label for unlabeled (test) data. 
# Here we assume (1) all training data are labeled and 
#                (2) all data have same # of sectors
NSectors = dataFull[0]['strainMat'].shape[-2]
isFakeTestLabel = False
for datum in dataFull:
    if 'TOS' not in datum.keys():
        datum['TOS'] = np.zeros((1, NSectors))
        datum['fakeTOS'] = True
        isFakeTestLabel = True

# Inverse TOS
if configs[0]['data'].get('Inverse TOS', False):
    print('inverse TOS!')
    for data in dataFull:        
        data['TOS'] = np.flip(data['TOS'])

# Train-test Split
if 'strainMat' in allDataTypes:
    NFramesMax = np.max([datum['strainMat'].shape[-1] for datum in dataFull])
elif 'dispField' in allDataTypes:
    NFramesMax = np.max([datum['dispField'].shape[2] for datum in dataFull])
NFramesMax = int(NFramesMax)


getPatientNameFromDataFilename = lambda dataFilename: dataFilename.split('Jerry/')[1].split('/mat')[0]
patientValidNames = [getPatientNameFromDataFilename(dataFilename) for dataFilename in dataFilenamesValid]
patientUniqueNames = list(dict.fromkeys(patientValidNames).keys())
NPatients = len(patientUniqueNames)

# from itertools import groupby
# sliceCountMax = max([len(list(group)) for key, group in groupby(patientValidNames)])

# Add Patinent Number
for datum in dataFull:
    datum['patientNo'] = patientUniqueNames.index(getPatientNameFromDataFilename(datum['dataFilename']))

indices4SlicesFixedBySliceTr, indices4SlicesFixedBySliceTe     = trainTestSplit(len(dataFull), 0.8, orgBy = 'slice', splBy = 'slice', random = 'fixed')
indices4SlicesRandBySliceTr, indices4SlicesRandBySliceTe       = trainTestSplit(len(dataFull), 0.8, orgBy = 'slice', splBy = 'slice', random = 'random')
indices4SlicesMixedByPatientTr, indices4SlicesMixedByPatientTe = trainTestSplit(len(dataFull), 0.8, orgBy = 'slice', splBy = 'slice', random = 'mixed')
indices4SlicesFixedByPatientTr, indices4SlicesFixedByPatientTe = trainTestSplit(len(dataFull), 0.8, orgBy = 'slice', splBy = 'patient', random = 'fixed', dataIDs = patientValidNames)
indices4SlicesRandByPatientTr, indices4SlicesRandByPatientTe   = trainTestSplit(len(dataFull), 0.8, orgBy = 'slice', splBy = 'patient', random = 'random', dataIDs = patientValidNames)
indices4PatientsFixedTr, indices4PatientsFixedTe = trainTestSplit(NPatients, 0.8, orgBy = 'patient', splBy = 'patient', random = 'fixed')
indices4PatientsRandTr, indices4PatientsRandTe   = trainTestSplit(NPatients, 0.8, orgBy = 'patient', splBy = 'patient', random = 'random')

ifSliceHaveTOS = np.array([not datum.get('fakeTOS', False) for datum in dataFull])
ifPatitentHaveTOS = [True] * NPatients
for datum in dataFull:
    ifPatitentHaveTOS[datum['patientNo']] *= not datum.get('fakeTOS', False)
ifPatitentHaveTOS = np.array(ifPatitentHaveTOS, dtype=np.bool)

indices4SlicesAlllabeledBySliceTr, indices4SlicesAlllabeledBySliceTe = trainTestSplit(len(dataFull), orgBy = 'slice', splBy = 'slice', random = 'allLabeled', ifDataHaveTOS = ifSliceHaveTOS)
indices4PatientsAlllabeledTr, indices4PatientsAlllabeledTe           = trainTestSplit(NPatients, orgBy = 'patient', splBy = 'patient', random = 'allLabeled', ifDataHaveTOS = ifPatitentHaveTOS)



#%% 3. Create Experiment Results Folder
if saveResults:
    expgPath, expgName = createExpFolder(resultPath, configName, create_subFolder=False, addDate = True)    
    
#%% 4. Run Experiments
for expIdx, config in enumerate(configs):
    #%% 1. Extrac and Save config
    print(f'Start config {expIdx} / {len(configs)-1}')
    print(config)
    expFoldername = config['name'] if len(config['name'])>0 else f'idx-{expIdx + startIdx}'
    expPath, expName = createExpFolder(expgPath, expFoldername, create_subFolder=False, addDate = False) 
    
    saveConfig2Json(config, expPath + 'config.json')
    # https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
    with open(expPath + config['name'] + '-config.pkl', 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
    
    # dataConfig = config['data']
    # training_config = config['training']
    # net_config = config['net']
    # loss_config = config['loss']
    
    
    #%% 2. Extract Used Data
    ifMergeDataByPatient = config['data'].get('mergeAllSlices', False)
    # Train-Test Split
    # It's better to merge training and test seperately, since test data maybe different (e.g. no label)
    if not ifMergeDataByPatient:
        if config['data']['train_test_split'] == 'random':
            train_indices = indices4SlicesRandBySliceTr
            test_indices = indices4SlicesRandBySliceTe
        elif config['data']['train_test_split'] == 'fixedPatient':
            train_indices = indices4SlicesFixedByPatientTr
            test_indices = indices4SlicesFixedByPatientTe
        elif config['data']['train_test_split'] == 'randomPatient':
            train_indices = indices4SlicesRandByPatientTr
            test_indices = indices4SlicesRandByPatientTe
        elif config['data']['train_test_split'] == 'mixedPatient':
            train_indices = indices4SlicesMixedByPatientTr
            test_indices = indices4SlicesMixedByPatientTe
        elif config['data']['train_test_split'] == 'allLabeled':
            train_indices = indices4SlicesAlllabeledBySliceTr
            test_indices = indices4SlicesAlllabeledBySliceTe
    else:
        if config['data']['train_test_split'] == 'random':
            train_indices = indices4PatientsRandTr
            test_indices = indices4PatientsRandTe
        elif config['data']['train_test_split'] == 'fixed':
            train_indices = indices4PatientsFixedTr
            test_indices = indices4PatientsFixedTe
        elif config['data']['train_test_split'] == 'allLabeled':
            train_indices = indices4PatientsAlllabeledTr
            test_indices = indices4PatientsAlllabeledTe
    
    # -------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------- #
    # FOR ISBI PAPER:
    # MOVE SET01-CT01 and SET02-EC03 to test set since the 3D TOS maps looks fine
    # and move two from tes set to training set
    def move_patient_data_fromTrain_toTest(train_indices, test_indices, patient_names = []):
        train_indices_new = train_indices.copy()
        test_indices_new = test_indices.copy()
                
        # # Find the names of patient in original test set    
        names_4_test_Patients = list(dict.fromkeys([patientValidNames[idx] for idx in test_indices_new]).keys())
        
        for patient_name in patient_names:
            if patient_name in names_4_test_Patients:
                raise ValueError(f'Patient {patient_name} already in test set')
        
        for patient_idx, patient_name in enumerate(patient_names):            
            # Get the indices of slices of the patient
            indices_of_patient_inAll = [idx for idx, name in enumerate(patientValidNames) if name == patient_name]
            indices_of_patient_inTraining = [int(np.where(train_indices_new==patient_slice_index)[0][0]) for patient_slice_index in indices_of_patient_inAll]            
            
            # Get the indices of a patient in original test set
            indices_of_patient_toBeMovedToTraining = [idxInAll for idxInAll in test_indices if patientValidNames[idxInAll] == names_4_test_Patients[patient_idx]]
            
            # Move from test to training
            # print(test_indices_new)
            test_indices_new = np.delete(test_indices_new, [idx for idx, idxInAll in enumerate(indices_of_patient_toBeMovedToTraining)])
            # print(test_indices_new)
            train_indices_new = np.concatenate([train_indices_new, indices_of_patient_toBeMovedToTraining])
            
            # Move from training to test
            # print(train_indices_new)
            train_indices_new = np.delete(train_indices_new, indices_of_patient_inTraining)
            # print(train_indices_new)
            # print(test_indices_new, indices_of_patient_inAll)
            test_indices_new = np.concatenate([test_indices_new, indices_of_patient_inAll])
            # print(test_indices_new)
        return train_indices_new, test_indices_new
        
    
    # train_indices_new = train_indices.copy()
    # test_indices_new = test_indices.copy()
    
    # # Find indices for the two patients
    # indices_4_SET01_CT01 = [idx for idx, name in enumerate(patientValidNames) if name == 'SET01\CT01']
    # # indices_4_SET02_EC03 = [idx for idx, name in enumerate(patientValidNames) if name == 'SET02\EC03']
    # indices_4_SET01_CT19 = [idx for idx, name in enumerate(patientValidNames) if name == 'SET01\CT19']
    
    # # Find the names of two other patient in the test set    
    # # indices_4_test_Patients = [idx for idx in range(len(patientValidNames)) if idx in test_indices]
    # names_4_test_Patients = list(dict.fromkeys([patientValidNames[idx] for idx in test_indices]).keys())
    # # if names_4_test_Patients[0] not in ['SET01\CT01','SET02\EC03']:
    # if names_4_test_Patients[0] not in ['SET01\CT01','SET01\CT19']:
    #     indices_4_test_Patients_0 = [idx for idx in test_indices if patientValidNames[idx] == names_4_test_Patients[0]]        
    # # if names_4_test_Patients[1] not in ['SET01\CT01','SET02\EC03']:
    # if names_4_test_Patients[1] not in ['SET01\CT01','SET01\CT19']:
    #     indices_4_test_Patients_1 = [idx for idx in test_indices if patientValidNames[idx] == names_4_test_Patients[1]]
            
    # # Move first two in test to train
    # test_indices_new = np.delete(test_indices_new, [idx for idx, idxInAll in enumerate(indices_4_test_Patients_0)])
    # test_indices_new = np.delete(test_indices_new, [idx for idx, idxInAll in enumerate(indices_4_test_Patients_1)])
    # train_indices_new = np.concatenate([train_indices_new, indices_4_test_Patients_0, indices_4_test_Patients_1])
    
    # # Move the two from train to test
    # train_indices_new = np.delete(train_indices_new, [idx for idx, idxInAll in enumerate(indices_4_SET01_CT01)])
    # #train_indices_new = np.delete(train_indices_new, [idx for idx, idxInAll in enumerate(indices_4_SET02_EC03)])
    # train_indices_new = np.delete(train_indices_new, [idx for idx, idxInAll in enumerate(indices_4_SET01_CT19)])
    # #test_indices_new = np.concatenate([test_indices_new, indices_4_SET01_CT01, indices_4_SET02_EC03])
    # test_indices_new = np.concatenate([test_indices_new, indices_4_SET01_CT01, indices_4_SET01_CT19])
    
    # train_indices = train_indices_new
    # test_indices = test_indices_new
    # test_indices = np.array([81, 82, 83, 84, 85,92, 93, 94, 95, 86, 87, 88, 89, 90, 91])
    train_indices, test_indices = move_patient_data_fromTrain_toTest(train_indices, test_indices, ['SET01\\CT01','SET02\\EC03'])
    
    # -------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------- #
    
    config['data']['train_indices'] = list(train_indices)
    config['data']['test_indices'] = list(test_indices)
    

    if ifMergeDataByPatient:
        # strainMat: [1, 1, NSectors, NFrames] -> [1, NSlices, NSectors, NFrames]
        dataFullUsed = mergeDataByPatient(dataFull, patientValidNames, reorderByLoc = True)
        NSlicesMax = np.max([datum['strainMat'].shape[1] for datum in dataFullUsed])
    else:
        dataFullUsed = dataFull
        
    dataTr = [dataFullUsed[idx] for idx in train_indices]
    dataTe = [dataFullUsed[idx] for idx in test_indices]
    for datum in dataTr: datum['isTraining'] = True
    for datum in dataTe: datum['isTraining'] = False
        
    #%% 3. Pre-processing and Augmentation
    # Unify Shape
    from utils.cardiacUtils import changeNFrames, changeNSlices
    for dataType in set([config['data']['inputType'],config['data']['outputType'], 'strainMatSVD', 'TOS']):
        for datum in dataTr + dataTe: changeNFrames(datum, NFramesMax, dataType)
        if config['data'].get('mergeAllSlices', False):
            for datum in dataTr + dataTe: changeNSlices(datum, NSlicesMax, dataType, config['data'].get('mergeAllSlices', False))
    
    # Truncate zeros to speed up training
    yMin, yMax = 30, 98
    xMin, xMax = 30, 98
    if config['data']['inputType'] in ['dispField']:
        config['data']['truncate']={'yMin':xMin,'yMax':xMax,'xMin':xMin,'xMax':xMax}
        for datum in dataTr + dataTe:
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
        dataTr = augmentation(dataTr.copy(), augConfig)
        # inputDataTr, labelDataTr = augmentation(inputDataTrRaw, labelDataTrRaw, augConfig)
        
        
    # else:
    #     inputDataTr, labelDataTr = inputDataTrRaw, labelDataTrRaw

    #%% 4. Prepare input data
    # Make 3D data: 
    # strainMat: [N, NSlices, NSectors, NFrames] -> [N, 1, NSlices, NSectors, NFrames]
    # TOS      : [N, NSlice, 1, NSectors] -> [N, 1, NSlice, 1, NSectors]
    if config['data']['inputType'] in ['strainMat', 'strainMatSVD']:
        if ifMergeDataByPatient:
            for datum in dataTr + dataTe:
                datum[config['data']['inputType']] = datum[config['data']['inputType']][:,None,:,:,:]
            
    if config['data']['outputType'] in ['TOS']:
        if ifMergeDataByPatient:
            for datum in dataTr + dataTe:
                datum[config['data']['outputType']] = datum[config['data']['outputType']][:,None,:,:,:]
    
    inputDataTr = np.concatenate([datum[config['data']['inputType']] for datum in dataTr], axis=0)
    inputDataTe = np.concatenate([datum[config['data']['inputType']] for datum in dataTe], axis=0)
    labelDataTr = np.concatenate([datum[config['data']['outputType']] for datum in dataTr], axis=0)
    labelDataTe = np.concatenate([datum[config['data']['outputType']] for datum in dataTe], axis=0)
    # if config['data'].get('mergeAllSlices', False):
    #     # Make 3D data: 
    #     # strainMat: [N, NSlices, NSectors, NFrames] -> [N, 1, NSlices, NSectors, NFrames]
    #     # TOS      : [N, NSlice, 1, NSectors] -> [N, 1, NSlice, 1, NSectors]
    #     inputDataTr = inputDataTr[:,None,:,:,:]
    #     inputDataTe = inputDataTe[:,None,:,:,:]
    #     labelDataTr = labelDataTr[:,None,:,:,:]
    #     labelDataTe = labelDataTe[:,None,:,:,:]
    device = torch.device(f"cuda:{gpuIdx}" if torch.cuda.is_available() else "cpu")
    if config['loss'] in ['contrastive']:        
        from modules.DataSet import DataSetContrastive
        inputPatientNosTr = np.array([datum['patientNo'] for datum in dataTr])
        inputPatientNosTe = np.array([datum['patientNo'] for datum in dataTe])
        training_dataset = DataSetContrastive(inputDataTr, inputPatientNosTr, device)
        test_dataset = DataSetContrastive(inputDataTe, inputPatientNosTe, device)
    else:
        from modules.DataSet import DataSet2D        
        # training_dataset = DataSet2D(imgs = inputDataTr, labels = labelDataTr, labelmasks = labelmasksTr, transform=None, device = device)
        # test_dataset = None if isFakeTestLabel else DataSet2D(imgs = inputDataTe, labels = labelDataTe, labelmasks = labelmasksTe, transform=None, device = device)
        training_dataset = DataSet2D(imgs = inputDataTr, labels = labelDataTr, transform=None, device = device)
        test_dataset = None if isFakeTestLabel else DataSet2D(imgs = inputDataTe, labels = labelDataTe, transform=None, device = device)

    #%% 4. Get Network
    # net_config['paras']['n_conv_layers'] = 4
    # net_config['paras']['n_conv_channels'] = 16
    # config['loss'] = {'name': 'MSE_POWER', 'para':{'power':2}}
    from modules.netModule import NetModule
    from modules.strainNets import getNetwork
    net_config = config['net']
    net_config['paras']['n_frames'] = NFramesMax
    net_config['inputType'] = config['data']['inputType']
    net_config['inputShape'] = inputDataTr.shape
    net_config['outputType'] = config['data']['outputType']    
    net_config['n_slices'] = inputDataTr.shape[2] if ifMergeDataByPatient else None
    net_config['mergeAllSlices'] = ifMergeDataByPatient

    net = NetModule(getNetwork(net_config), config['loss'], device)   
    
    #%% 5. Training
    if loadExistingConfig:
        # netFile = '../COPIED-2020-Jun-29-useData201-scarFree-strainMat-fixedSpitbyPat/idx-260/idx-260-strainMatSVD-AugShift-LR1.00E-04-CV3-Ch8-net.pth'
        # netFile = './temp/idx-1-strainMat-AugNone-LR1.00E-04-CV5-Ch4-net.pth'
        netFile = './temp/idx-260-strainMatSVD-AugShift-LR1.00E-04-CV3-Ch8-net-retrain.pth'
        net.load(netFile, map_location={'cuda:1': f'cuda:{gpuIdx}','cuda:2': f'cuda:{gpuIdx}','cuda:3': f'cuda:{gpuIdx}'})
    else:
        if debug:
            config['training']['learning_rate'] = 1e-4
            config['training']['epochs_num'] = 50
            config['training']['batch_size'] = 10
        if config['training']['batch_size'] == 'half':
            config['training']['batch_size'] = max(len(dataTr)//2, 10)
        # ---------------------------------------------- #
        # ------------ FOR ISBI PAPER ------------------ #
        # ---------------------------------------------- #
        # net.continueTraining = True
        config['training']['batch_size'] = max(len(dataTr)//5, 10)
        # config['training']['learning_rate'] = 5e-4
        config['training']['epochs_num'] = 3000
        loss_history, validLoss_history, past_time = net.train(training_dataset=training_dataset, training_config = config['training'], valid_dataset = test_dataset, expPath = None)
        
        
    
    #%% 6. Validate Model    
    # Get Prediction
    # import time
    for datum in dataTr + dataTe:
        # datum['dataFilename'][datum['isOriSlice'] == False] = 'Interpreted'
        # for sidx in range(NSlicesMax): datum['dataFilename']
        datum['predRaw'], datum['loss'] = net.pred(datum[config['data']['inputType']], labels = datum[config['data']['outputType']], avg = True)
    
    # Get prediction time
    # import time
    # predTimes = np.zeros(len(dataTe)*3)
    # for datumIdx, datum in enumerate(dataTe+dataTe+dataTe):
    #     # datum['dataFilename'][datum['isOriSlice'] == False] = 'Interpreted'
    #     # for sidx in range(NSlicesMax): datum['dataFilename']
    #     predTimes[datumIdx] = time.time()
    #     _, _ = net.pred(datum[config['data']['inputType']], labels = None, avg = True)
    #     predTimes[datumIdx] = time.time() - predTimes[datumIdx]
    # print(predTimes)
    # print(np.mean(predTimes))
    # predTrRaw, lossTr = net.pred(inputDataTr, labels = labelDataTr, avg = True)
    # predTeRaw, lossTe = net.pred(inputDataTe, labels = labelDataTe, avg = True)
    
    # Test augmentation
    # for datum in dataTe:
    #     datumAuged = augmentation(augConfig)
    # dataTeAug = augmentation(copy.deepcopy(dataTe), augConfig)
    
    # for datum in dataTeAug:
    #     # datum['dataFilename'][datum['isOriSlice'] == False] = 'Interpreted'
    #     # for sidx in range(NSlicesMax): datum['dataFilename']
    #     datum['predRaw'], datum['loss'] = net.pred(datum[config['data']['inputType']], labels = datum[config['data']['outputType']], avg = True)
    
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
    
    # Append active Contour Results
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    activeContourPath = 'C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Matlab_utils\\active_contour\\TOSs\\'
    import scipy.io as sio
    from os import listdir
    from os.path import isfile, join
    activeContourFiles = [f for f in listdir(activeContourPath) if isfile(join(activeContourPath, f))]
    for activeContourFilename in activeContourFiles:
        activeContourData = sio.loadmat(activeContourPath + activeContourFilename)
        activeContourSet, activeContourPatient = activeContourData['patientName'][0].split('-')
        activeContourSlice = activeContourData['sliceName'][0]
        # print(activeContourSet, activeContourPatient,activeContourSlice)
        idxInDataTe = [idx for idx in range(len(dataTe)) if activeContourSet in dataTe[idx]['dataFilename'] 
                       and activeContourPatient in dataTe[idx]['dataFilename'] 
                       and activeContourSlice in dataTe[idx]['dataFilename']][0]
        # print(idxInDataTe)
        dataTe[idxInDataTe]['activeContourXs'] = activeContourData['xs']
        dataTe[idxInDataTe]['activeContourTime'] = activeContourData['time']
        dataTe[idxInDataTe]['activeContourIter'] = activeContourData['iter']
        # print(dataTe[idxInDataTe].keys())
    
    # Save Figures to file
    if debug:
        # savePath = './temp/pdfs/'
        savePath = './temp/imgs_leakyRELU/'
        # savePath = './temp/pdfs_truncated/'
        # savePath = './temp/imgs_useSET01CT19/'
    else:
        savePath = expPath
    if saveResults:
        from utils.cardiacUtils import getSliceName as filename2
        
        from utils.io import saveLossInfo
        # saveLossInfo(predTe, labelDataTe, savePath)
        saveSubplot = False
        truncate = 30
        
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
                              + '\n'+ filename2(sliceIDs[sidx]) \
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
            for patientIdx, patientName in enumerate(['SET01\\CT01', 'SET02\\EC03','SET03\\EC21']):
            # for patientIdx, patientName in enumerate(patientUniqueNames):
                # dataOfPatient, dataIdxs = zip(*[(datum, datumIdx) for (datumIdx, datum) in enumerate(dataTr + dataTe) if patientName in datum.get('dataFilename', 'NONE') and not datum.get('isAugmented', False)])
                # dataOfPatient, dataIdxs = zip(*[(datum, datumIdx) for (datumIdx, datum) in enumerate(dataTr + dataTe) if patientName in datum.get('dataFilename', 'NONE')])
                dataOfPatient, dataIdxs = zip(*[(datum, datumIdx) for (datumIdx, datum) in enumerate(dataTe) if patientName in datum.get('dataFilename', 'NONE')])
                sliceLocs = [datum['SequenceInfo'] for datum in dataOfPatient]
                dataOfPatientSorted = [dataOfPatient[idx] for idx in np.argsort(sliceLocs)]
                dataIdxsSorted = [dataIdxs[idx] for idx in np.argsort(sliceLocs)]
                sliceLocsSorted = [sliceLocs[idx] for idx in np.argsort(sliceLocs)]
                
                if saveSubplot:
                    strainMats = np.squeeze(np.concatenate([datum['strainMatSVD'] for datum in dataOfPatientSorted]))
                    TOSs = np.concatenate([datum['TOS'] for datum in dataOfPatientSorted])
                    preds = np.concatenate([datum['pred'] for datum in dataOfPatientSorted])
                    MSEs = np.linalg.norm(TOSs - preds, axis=-1)
                    title = '↑ ' + config['name'] + '\n' + configName
                    subtitles = [(f'train {dataIdxs[sidx]}' if dataIdxsSorted[sidx]<len(dataTr) else f'test {dataIdxs[sidx] - len(dataTr)}') \
                                 + f'\n {filename2(dataOfPatientSorted[sidx]["dataFilename"])}' \
                                 + f'\n SliceLoc = {sliceLocsSorted[sidx]:.2f}'
                                 + f'\n MSE={MSEs[sidx]:.2f}'
                                 for sidx in range(len(dataOfPatientSorted))]
                    saveFilename = savePath + f'result_{patientIdx}'
                    saveStrainDataFig2File(strainMats, 
                            [TOSs, preds],
                            saveFilename, 
                            legends = ['Ground Truth', 'Ours'], 
                            title = title, subtitles = subtitles,
                            inverseTOS = not config['data'].get('Inverse TOS', False))
                else:
                    for sliceDataOfPatient in dataOfPatientSorted:
                        sliceTOS = None if sliceDataOfPatient.get('fakeTOS',False) else np.squeeze(sliceDataOfPatient['TOS'])
                        saveFilename = savePath + ('train' if sliceDataOfPatient['isTraining'] else 'test') + \
                                        '-' + filename2(sliceDataOfPatient['dataFilename']) + '.png'
                        saveStrainDataFig2File(np.squeeze(sliceDataOfPatient['strainMatSVD']),
                                                [sliceTOS, np.squeeze(sliceDataOfPatient['activeContourXs']), np.squeeze(sliceDataOfPatient['pred'])],
                                                saveFilename, legends = ['Ground Truth', 'Active Contour', 'Ours'],
                                               # [sliceTOS, np.squeeze(sliceDataOfPatient['pred'])],
                                               #  saveFilename, legends = ['Ground Truth', 'Ours'],
                                               title = None, subtitles = None,
                                               inverseTOS = not config['data'].get('Inverse TOS', False),
                                               markPeak = False, truncate = None)
                
    config['performance'] = {
        'lossTrBatch': loss_history[-1],
        'lossTr': sum([datum['loss'] for datum in dataTr]),
        'lossTe': sum([datum['loss'] for datum in dataTe])
        }
    
    net.saveLossHistories([loss_history, validLoss_history], savePath + 'losses-retrain.png', 
                          report_epochs_num=config['training']['report_per_epochs'], 
                          legends=['training loss', 'test loss'])
    
    net.save(savePath + config['name'] + '-net-retrain.pth')
    
    saveConfig2Json(config, savePath + 'config.json')
    # https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
    with open(savePath + config['name'] + '-config-retrain.pkl', 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

if saveResults:
    with open(expgPath + 'configs.pkl', 'wb') as f:
        pickle.dump(configs, f, pickle.HIGHEST_PROTOCOL)
        
#%% Rebuild 3D TOS Map
data_SET01_CT01 = [datum for datum in dataTe if 'SET01\\CT01' in datum['dataFilename']]
data_SET02_EC03 = [datum for datum in dataTe if 'SET02\\EC03' in datum['dataFilename']]
data_SET03_EC21 = [datum for datum in dataTe if 'SET03\\EC21' in datum['dataFilename']]
from utils.TOS3DPlotInterpFunc import TOS3DPlotInterp

data_SET01_CT01_min = np.min(np.min([sliceData['pred'] for sliceData in data_SET01_CT01]))
data_SET01_CT01_max = np.max(np.max([sliceData['pred'] for sliceData in data_SET01_CT01]))
data_SET02_EC03_min = np.min(np.min([sliceData['pred'] for sliceData in data_SET02_EC03]))
data_SET02_EC03_max = np.max(np.max([sliceData['pred'] for sliceData in data_SET02_EC03]))

from scipy.interpolate import interp1d
def interpTOS(TOS, layerLength, layerNum):
    # layerLength = sum(dataTe[0]['AnalysisFv'].layerid == 1)
    # layerNum = len(np.unique(dataTe[0]['AnalysisFv'].layerid))
    fakeCircularRepeatNum = 3
    fillInterval = np.floor(layerLength / np.size(TOS));
    #TOSFillIndices = fillInterval*(0:len(TOS)*fakeCircularRepeatNum-1) + 1;
    TOSFillIndices = fillInterval*np.arange(np.size(TOS)*fakeCircularRepeatNum) + 1;
    # arrayInterpIndicesFakeCircular = 1:layerLength*fakeCircularRepeatNum;
    arrayInterpIndicesFakeCircular = np.arange(layerLength*fakeCircularRepeatNum) +1

    f = interp1d(TOSFillIndices, 
        np.squeeze(np.tile(TOS, fakeCircularRepeatNum)),
        'cubic',
        fill_value = 'extrapolate');
    arrayInterpFakeCircular = f(arrayInterpIndicesFakeCircular)

    TOSInterpedMidLayer = arrayInterpFakeCircular[
        int(np.floor(fakeCircularRepeatNum/2)*layerLength):
        int(np.floor(fakeCircularRepeatNum/2)*layerLength+layerLength)]

    TOSInterped = np.tile(TOSInterpedMidLayer, [1, layerNum]);
    
    return TOSInterped
for sliceData in data_SET01_CT01:
    sliceData['TOSInterpolatedNew'] = interpTOS(sliceData['TOS'], sum(sliceData['AnalysisFv'].layerid == 1), len(np.unique(sliceData['AnalysisFv'].layerid)))
    sliceData['TOSPredInterpolated'] = interpTOS(sliceData['pred'], sum(sliceData['AnalysisFv'].layerid == 1), len(np.unique(sliceData['AnalysisFv'].layerid)))

for sliceData in data_SET02_EC03:
    sliceData['TOSInterpolatedNew'] = interpTOS(sliceData['TOS'], sum(sliceData['AnalysisFv'].layerid == 1), len(np.unique(sliceData['AnalysisFv'].layerid)))
    sliceData['TOSPredInterpolated'] = interpTOS(sliceData['pred'], sum(sliceData['AnalysisFv'].layerid == 1), len(np.unique(sliceData['AnalysisFv'].layerid)))

for sliceData in data_SET03_EC21:
    sliceData['TOSInterpolatedNew'] = interpTOS(sliceData['TOS'], sum(sliceData['AnalysisFv'].layerid == 1), len(np.unique(sliceData['AnalysisFv'].layerid)))
    sliceData['TOSPredInterpolated'] = interpTOS(sliceData['pred'], sum(sliceData['AnalysisFv'].layerid == 1), len(np.unique(sliceData['AnalysisFv'].layerid)))    

# interpMethod = 'slinear'
interpMethod = 'quadratic'
# TOS3DPlotInterp(data_SET01_CT01, toshow = 'TOSInterploated', interpMethod = interpMethod)
# TOS3DPlotInterp(data_SET01_CT01, toshow = 'TOSInterpolatedNew', interpMethod = interpMethod, azim = 75, vmin = data_SET01_CT01_min/17, vmax = data_SET01_CT01_max/17)
# TOS3DPlotInterp(data_SET01_CT01, toshow = 'TOSPredInterpolated', interpMethod = interpMethod, azim = 75, vmin = data_SET01_CT01_min/17, vmax = data_SET01_CT01_max/17)
TOS3DPlotInterp(data_SET01_CT01, toshow = 'TOSInterpolatedNew', interpMethod = interpMethod, azim = 75, vmin=None, vmax = None)
# TOS3DPlotInterp(data_SET01_CT01, toshow = 'TOSPredInterpolated', interpMethod = interpMethod, azim = 75, vmin=None, vmax = None)

# TOS3DPlotInterp(data_SET02_EC03, toshow = 'TOSInterploated', interpMethod = interpMethod)
# TOS3DPlotInterp(data_SET02_EC03, toshow = 'TOSInterpolatedNew', interpolate = False, vmin = 15, vmax = 70, interpMethod = interpMethod)
# TOS3DPlotInterp(data_SET02_EC03, toshow = 'TOSPredInterpolated', interpolate = False, vmin = 15, vmax = 70, interpMethod = interpMethod)
# TOS3DPlotInterp(data_SET02_EC03, toshow = 'TOSInterpolatedNew', interpolate = True, vmin = None, vmax = None, interpMethod = interpMethod, azim = -30)
TOS3DPlotInterp(data_SET02_EC03, toshow = 'TOSPredInterpolated', interpolate = True, vmin = None, vmax = None, interpMethod = interpMethod, azim = -30)
# TOS3DPlotInterp(data_SET02_EC03, toshow = 'TOSInterpolatedNew', interpolate = True, vmin = data_SET01_CT01_min/17, vmax = data_SET01_CT01_max/17, interpMethod = interpMethod, azim = -30)
# TOS3DPlotInterp(data_SET02_EC03, toshow = 'TOSPredInterpolated', interpolate = True, vmin = data_SET01_CT01_min/17, vmax = data_SET01_CT01_max/17, interpMethod = interpMethod, azim = -30)

TOS3DPlotInterp(data_SET03_EC21, toshow = 'TOSInterpolatedNew', interpMethod = interpMethod, azim = 75, vmin=1, vmax = None)
TOS3DPlotInterp(data_SET03_EC21, toshow = 'TOSPredInterpolated', interpMethod = interpMethod, azim = 75, vmin=1, vmax = None)

#%% Plot the pred_peak_sectorID - GT_peak_sectorID scatter plot
peakLocsPred = np.array([np.argmax(sliceData['pred']) for sliceData in dataTe]).reshape([1,-1])
peakLocsGT = np.array([np.argmax(sliceData['TOS']) for sliceData in dataTe]).reshape([1,-1])
peakLocsUniqVecs = np.unique(np.concatenate([peakLocsPred, peakLocsGT], axis=0), axis=1)
peakLocsUniqCount = np.array([np.sum((peakLocsPred==peakLocsUniqVecs[0,vecIdx])*(peakLocsGT==peakLocsUniqVecs[1,vecIdx])) for vecIdx in range(peakLocsUniqVecs.shape[1])])

plt.figure()
#plt.scatter(peakLocsPred, peakLocsGT, s = (30 * np.random.rand(len(peakLocsPred)))**2, c = np.random.rand(len(peakLocsPred)), alpha=0.5)
# plt.scatter(peakLocsUniqVecs[0,:], peakLocsUniqVecs[1,:], s = peakLocsUniqCount*300, c = np.random.rand(peakLocsUniqVecs.shape[1]), alpha=0.5)
plt.scatter(peakLocsPred, peakLocsGT)
# plt.xlim([1,18])
# plt.ylim([1,18])

plt.figure()
plt.hist(np.abs(peakLocsGT - peakLocsPred))

#%% Plot the pred_peak_sectorID - GT_peak_sectorID box plot
peakLocsPredAtLocs = []
for peakLocGT in np.unique(peakLocsGT):
    peakLocsPredAtLocs.append(peakLocsPred[peakLocsGT==peakLocGT])
    # peakLocsPredAtLocs.append(np.random.rand(50)*10)
    # peakLocsPredAtLocs.append(np.concatenate([peakLocsPred[peakLocsGT==peakLocGT], np.random.rand(50)*10]))
fig7, ax7 = plt.subplots()
# ax7.set_title('Multiple Samples with Different sizes')
ax7.boxplot(peakLocsPredAtLocs)
# ax7.set_xticks([0,1,2],[str(sectorID) for sectorID in np.unique(peakLocsGT)])
ax7.set_xticklabels([str(sectorID) for sectorID in np.unique(peakLocsGT)])
ax7.set_xlabel('GT Peak Segment ID')
ax7.set_ylabel('Predicted Segment ID')
# ax7.set_xticks
plt.show()

#%% Plot the Pred-GT peak location difference histogram
# https://matplotlib.org/3.3.2/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
peakLocsDiffs = np.abs(peakLocsPred - peakLocsGT)
peakLocsDiffsUnique, peakLocsDiffsUniqueCount = np.unique(peakLocsDiffs, return_counts = True)
# n, bins, patches = plt.hist(np.squeeze(peakLocsDiffs), density=True, facecolor='g', alpha=0.75)
fig, ax = plt.subplots()
rects = ax.bar(peakLocsDiffsUnique, peakLocsDiffsUniqueCount / np.size(peakLocsDiffs))
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['0', '1', '2'])
ax.set_xlabel('Distance between Ground Truth and Estimated Peak Location')
ax.set_ylabel('Fraction')

# data = np.random.randint(0, 10, 1000)

# bins = np.arange(11) - 0.5
# plt.hist(data, bins)
# plt.xticks(range(10))
# plt.xlim([-1, 10])

plt.show()


#%% Show displacement & strain for ther background section
plt.close('all')
import scipy.io as sio

# bgSlice = dataFull[1].copy()
# bgSliceMat = sio.loadmat('C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET01\\CT01\\mat\\Mid1.mat', struct_as_record=False, squeeze_me = True)
# bgSliceMat = sio.loadmat('C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\paper\\ISBI\\figures\\auto.1_LV_base.mat', struct_as_record=False, squeeze_me = True)
bgSlice = dataFull[2].copy()
bgSliceMat = sio.loadmat('C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET01\\CT01\\mat\\Mid2.mat', struct_as_record=False, squeeze_me = True)
# bgSlice = dataFull[78].copy()
# bgSliceMat = sio.loadmat('C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET02\\EC03\\mat\\Base.mat', struct_as_record=False, squeeze_me = True)
# bgSlice = dataFull[87].copy()
# bgSliceMat = sio.loadmat('C:\\Users\\remus\\OneDrive\\Documents\\Study\\Researches\\Projects\\cardiac\\Dataset\\CRT_TOS_Data_Jerry\\SET03\\EC19\\mat\\SL2.mat', struct_as_record=False, squeeze_me = True)
bgSlice['ImageInfo'] = bgSliceMat['ImageInfo']
# for bgFrame in np.arange(5, 50, 5):
bgFrame = 5
bgDispX = np.squeeze(bgSlice['dispField'][:,0,bgFrame,:,:])
bgDispY = np.squeeze(bgSlice['dispField'][:,1,bgFrame,:,:])
bgMask = ~np.isnan(np.squeeze(bgSlice['ImageInfo'].Xunwrap[:,:,bgFrame]))
bgDispX[bgMask==0] = np.nan
bgDispY[bgMask==0] = np.nan

plt.figure()
plt.tight_layout()
plt.axis('off')
# plt.imshow(np.squeeze(bgSlice['ImageInfo'].Mag[45:85,40:85,bgFrame]), cmap='gray')
# bgMag = np.rot90(np.squeeze(bgSlice['ImageInfo'].Mag))
bgMag = np.squeeze(bgSlice['ImageInfo'].Mag)
plt.imshow(bgMag[45:85,40:85,bgFrame], cmap='gray')
# plt.pcolor(bgMag[45:85,40:85,bgFrame], cmap='gray')

plt.quiver(bgDispX[45:85,40:85],bgDispY[45:85,40:85], color = [1,1,0])

# bgContour0 = bgSliceMat['ROIInfo'].Contour[bgFrame,0]
# bgContour1 = bgSliceMat['ROIInfo'].Contour[bgFrame,1]
# # plt.plot(bgContour0[:,0]-40,bgContour0[:,1]-45)
# # plt.plot(bgContour1[:,0]-40,bgContour1[:,1]-45)
# plt.plot(bgContour0[:,1]-45,bgContour0[:,0]-40)
# plt.plot(bgContour1[:,1]-45,bgContour1[:,0]-40)

# bgSlice['ImageInfo'].Mag = (bgSlice['ImageInfo'].Mag - np.min(bgSlice['ImageInfo'].Mag)) / (np.max(bgSlice['ImageInfo'].Mag) - np.min(bgSlice['ImageInfo'].Mag))
# plt.imshow(np.squeeze(bgSlice['ImageInfo'].Mag[35:75,40:85,bgFrame]), cmap='gray')
# plt.quiver(bgDispX[35:75,40:85],bgDispY[35:75,40:85], color = [1,1,0])
# SECTORSHIFT = 14
SECTORSHIFT = 0
bgStraimImg = bgSlice['strainImg'][0,0,bgFrame,:,:]
bgStraimImg[bgMask==0] = np.nan
# bgStraimImg[bgStraimImg==0] = np.nan
plt.figure()
plt.imshow(bgStraimImg[45:85,40:85], cmap='jet', vmin = -0.2, vmax = 0.2)
# plt.imshow(bgStraimImg[35:75,40:85], cmap='jet', vmin = -0.2, vmax = 0.2)
# plt.colorbar(orientation="horizontal", pad=0.2)
plt.colorbar()
plt.axis('off')
plt.tight_layout()
# plt.pcolor(bgStraimImg[45:85,40:85], cmap='jet', vmin = -0.2, vmax = 0.2)

# plt.figure()
# plt.imshow(np.squeeze(bgSlice['strainMat']), cmap='jet', vmin = -0.2, vmax = 0.2)
# fig, axe = plt.subplots()
# axe.pcolor(np.squeeze(bgSlice['strainMat']), cmap='jet', vmin = -0.2, vmax = 0.2)

def visTOSs(TOSs, strainMat = None, legends = None, title = None):
    fig, axe = plt.subplots()
    plt.tight_layout()
    if strainMat is not None:
        matPlot = axe.pcolor(strainMat, cmap='jet', vmin = -0.2, vmax = 0.2)
    for idx, TOS in enumerate(TOSs):
        TOSGrid = np.flip((TOS / 17).flatten() - 0.5)
        line, = axe.plot(TOSGrid,np.arange(len(TOS))+0.5, linewidth = 3)
        if legends is not None:
            line.set_label(legends[idx])
        if title is not None:
            axe.set_title(title)
    if legends is not None:
        axe.legend()
    fig.colorbar(matPlot)
# visTOSs([np.squeeze(bgSlice['TOS'])],np.squeeze(bgSlice['strainMat']))
visTOSs([np.roll(np.squeeze(bgSlice['TOS']),-SECTORSHIFT)],np.roll(np.squeeze(bgSlice['strainMat']), SECTORSHIFT,axis=0))
plt.figure();plt.tight_layout();plt.plot(np.roll(np.squeeze(bgSlice['strainMat']), SECTORSHIFT,axis=0)[11,:], linewidth = 4)

# Compute and show segment ID image
# bgSegmentIDImg = np.zeros(bgDispX.shape)
bgSegmentFaceCenterXs = np.zeros(len(bgSlice['AnalysisFv'].faces))
bgSegmentFaceCenterYs = np.zeros(len(bgSlice['AnalysisFv'].faces))
for faceIdx, face in enumerate(bgSlice['AnalysisFv'].faces):
    faceXs = bgSlice['AnalysisFv'].vertices[face - 1,0]
    faceYs = bgSlice['AnalysisFv'].vertices[face - 1,1]
    #faceCenterX, faceCenterY = (np.ceil(np.mean(faceXs)), np.ceil(np.mean(faceYs)))
    #bgSegmentIDImg[int(faceCenterY), int(faceCenterX)] = bgSlice['AnalysisFv'].sectorid[faceIdx]
    bgSegmentFaceCenterXs[faceIdx] = np.mean(faceXs)
    bgSegmentFaceCenterYs[faceIdx] = np.mean(faceYs)
from scipy import interpolate
# f = interpolate.interp2d(bgSegmentFaceCenterXs, bgSegmentFaceCenterYs, bgSlice['AnalysisFv'].sectorid, kind='linear')
# bgSegmentIDImg = f(np.meshgrid(np.arange(bgDispX.shape[0]),np.arange(bgDispX.shape[1])))
fSegmentImg = interpolate.NearestNDInterpolator(np.vstack([bgSegmentFaceCenterXs,bgSegmentFaceCenterYs]).T, np.mod(bgSlice['AnalysisFv'].sectorid + 14,18))
bgXs, bgYs = np.meshgrid(np.arange(bgDispX.shape[0]),np.arange(bgDispX.shape[1]))
bgSegmentIDImg = fSegmentImg(np.vstack([bgXs.flatten(), bgYs.flatten()]).T).reshape(bgDispX.shape).astype(np.float).T
# fStrainImg = interpolate.interp2d(bgSegmentFaceCenterXs, bgSegmentFaceCenterYs, bgSlice['TransmuralStrainInfo'].Ecc.mid[], kind='linear')

bgSegmentIDImg[bgMask==0] = np.nan
plt.figure()
plt.tight_layout()
# plt.pcolor(bgSegmentIDImg[35:75,40:85], cmap='jet')
plt.imshow(bgSegmentIDImg[45:85,40:85], cmap='jet')
plt.axis('off')
plt.colorbar()

# plt.figure();plt.pcolor(bgMask[45:85,40:85])
# plt.figure();plt.imshow(bgMask[45:85,40:85], cmap='jet')

#%%
plt.close('all')
plt.figure()
plt.imshow(bgMag[:,:,bgFrame], cmap='gray')
bgContour0 = bgSliceMat['ROIInfo'].Contour[bgFrame,0]
bgContour1 = bgSliceMat['ROIInfo'].Contour[bgFrame,1]
plt.plot(bgContour0[:,0],bgContour0[:,1])
plt.plot(bgContour1[:,0],bgContour1[:,1])