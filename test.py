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

def loadConfig(filename):
    with open(filename, 'rb') as f:
        config = pickle.load(f) 
    return config
# configName = '../COPIED-2020-Jun-29-useData201-scarFree-strainMat-fixedSpitbyPat/idx-260/idx-260-strainMatSVD-AugShift-LR1.00E-04-CV3-Ch8-config.pkl'
configName = './temp/idx-260-strainMatSVD-AugShift-LR1.00E-04-CV3-Ch8-config-retrain.pkl'
config = loadConfig(configName)
config['outputType'] = 'TOS18_Jerry'
configs = [config]

# configs = [config for config in configs if config['net']['paras']['n_conv_layers']>3]
NConfigs = len(configs)
gpuIdx = 0

#%% 2. Prepare Data
print('Loading Data...')
dataFilename = 'dataFull-201-2020-12-23-Jerry.npy'
dataSaved = np.load(dataFilename, allow_pickle = True).item()
dataInfo = dataSaved['description']
dataFull = dataSaved['data']

# Exculde SETOLD since it's not used in original version
dataFull = [datum for datum in dataFull if 'SETOLD' not in datum['dataFilename']]
                    
# Exclude data from patients with scar
if configs[0]['data'].get('scarFree', False):
    dataFull = [datum for datum in dataFull if datum['hasScar'] == False]
    
allDataTypes = list(dataFull[0].keys())
dataFilenamesValid = list(datum['dataFilename'] for datum in dataFull)
print('Finished!')


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

# Inverse TOS to make sure the TOS matches the myocardium sectors
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
    # ISBI Final Version:
    # Change to CT11 and CT28 where AC is greatly misled
    from utils.trainTestSplit import move_patient_data_fromTrain_toTest        
    train_indices, test_indices = move_patient_data_fromTrain_toTest(train_indices, test_indices, patientValidNames, ['SET01\\CT11','SET02\\CT28'])
    
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
    device = torch.device(f"cuda:{gpuIdx}" if torch.cuda.is_available() else "cpu")
    if config['loss'] in ['contrastive']:        
        from modules.DataSet import DataSetContrastive
        inputPatientNosTr = np.array([datum['patientNo'] for datum in dataTr])
        inputPatientNosTe = np.array([datum['patientNo'] for datum in dataTe])
        training_dataset = DataSetContrastive(inputDataTr, inputPatientNosTr, device)
        test_dataset = DataSetContrastive(inputDataTe, inputPatientNosTe, device)
    else:
        from modules.DataSet import DataSet2D        
        training_dataset = DataSet2D(imgs = inputDataTr, labels = labelDataTr, transform=None, device = device)
        test_dataset = None if isFakeTestLabel else DataSet2D(imgs = inputDataTe, labels = labelDataTe, transform=None, device = device)

    #%% 4. Get Network
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
    netFile = './temp/idx-260-strainMatSVD-AugShift-LR1.00E-04-CV3-Ch8-net-retrain-ISBI-final.pth'
    net.load(netFile, map_location={'cuda:1': f'cuda:{gpuIdx}','cuda:2': f'cuda:{gpuIdx}','cuda:3': f'cuda:{gpuIdx}'})    
        
        
    
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
    
    # Save Figures to file
    if debug:
        pass
        # savePath = './temp/pdfs/'
        # savePath = './temp/imgs_leakyRELU/'
        # savePath = './temp/imgs_ISBI_final/'
        # savePath = './temp/pdfs_truncated/'
        # savePath = './temp/imgs_useSET01CT19/'
    else:
        # savePath = expPath
        savePath = './temp/imgs_ISBI_final_without_font3/pdf/'
    if saveResults:
        from utils.cardiacUtils import getSliceName as filename2
        
        from utils.io import saveLossInfo
        
        # plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
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
            # for patientIdx, patientName in enumerate(['SET01\\CT01', 'SET02\\EC03','SET03\\EC21']):
            for patientIdx, patientName in enumerate(['SET01\\CT11', 'SET02\\CT28','SET03\\EC21']):
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
                        sliceTOS = None if sliceDataOfPatient.get('fakeTOS',False) else np.squeeze(sliceDataOfPatient['TOS18_Jerry'])
                        sliceDataFilename = sliceDataOfPatient['dataFilename']
                        slicePatientName = sliceDataFilename.split('Data_Jerry/')[-1].split('/mat_processed')[0].replace('\\', '_')
                        sliceSliceName = sliceDataFilename.split('mat_processed_Jerry/')[-1].split('_processed')[0]
                        # saveFilename = savePath + ('train' if sliceDataOfPatient['isTraining'] else 'test') + \
                        #                 '-' + filename2(sliceDataOfPatient['dataFilename']) + '.png'
                        saveFilename = savePath + ('train' if sliceDataOfPatient['isTraining'] else 'test') + \
                                        '-' + slicePatientName + '_' + sliceSliceName + '.pdf'
                        saveStrainDataFig2File(np.squeeze(sliceDataOfPatient['strainMatSVD'])[:,:30],
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
    
    net.save(savePath + config['name'] + '-net-retrain-ISBI-final.pth')
    
    saveConfig2Json(config, savePath + 'config.json')
    # https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
    with open(savePath + config['name'] + '-config-retrain.pkl', 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

if saveResults:
    with open(expgPath + 'configs.pkl', 'wb') as f:
        pickle.dump(configs, f, pickle.HIGHEST_PROTOCOL)