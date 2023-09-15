# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:55:22 2020

@author: remus
"""

import numpy as np
from utils.extractCardiacData import trainTestSplit

def getIndices(indicesAll, ifMergeDataByPatient, randomness):
    if not ifMergeDataByPatient:
        if randomness == 'random':
            train_indices = indicesAll['indices4SlicesRandBySliceTr']
            test_indices = indicesAll['indices4SlicesRandBySliceTe']
        elif randomness == 'fixedPatient':
            train_indices = indicesAll['indices4SlicesFixedByPatientTr']
            test_indices = indicesAll['indices4SlicesFixedByPatientTe']
        elif randomness == 'randomPatient':
            train_indices = indicesAll['indices4SlicesRandByPatientTr']
            test_indices = indicesAll['indices4SlicesRandByPatientTe']
        elif randomness == 'mixedPatient':
            train_indices = indicesAll['indices4SlicesMixedByPatientTr']
            test_indices = indicesAll['indices4SlicesMixedByPatientTe']
        elif randomness == 'allLabeled':
            train_indices = indicesAll['indices4SlicesAlllabeledBySliceTr']
            test_indices = indicesAll['indices4SlicesAlllabeledBySliceTe']
    else:
        if randomness == 'random':
            train_indices = indicesAll['indices4PatientsRandTr']
            test_indices = indicesAll['indices4PatientsRandTe']
        elif randomness == 'fixed':
            train_indices = indicesAll['indices4PatientsFixedTr']
            test_indices = indicesAll['indices4PatientsFixedTe']
        elif randomness == 'allLabeled':
            train_indices = indicesAll['indices4PatientsAlllabeledTr']
            test_indices = indicesAll['indices4PatientsAlllabeledTe']
    return train_indices, test_indices

def getAllIndices(dataFull, inverseTOS = False):
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
    if inverseTOS:
        print('inverse TOS!')
        for data in dataFull:        
            data['TOS'] = np.flip(data['TOS'])
    
    # Train-test Split
    allDataTypes = list(dataFull[0].keys())
    if 'strainMat' in allDataTypes:
        NFramesMax = np.max([datum['strainMat'].shape[-1] for datum in dataFull])
    elif 'dispField' in allDataTypes:
        NFramesMax = np.max([datum['dispField'].shape[2] for datum in dataFull])
    NFramesMax = int(NFramesMax)
    
    dataFilenamesValid = list(datum['dataFilename'] for datum in dataFull)
    getPatientNameFromDataFilename = lambda dataFilename: dataFilename.split('Jerry/')[1].split('/mat')[0]
    patientValidNames = [getPatientNameFromDataFilename(dataFilename) for dataFilename in dataFilenamesValid]
    patientUniqueNames = list(dict.fromkeys(patientValidNames).keys())
    NPatients = len(patientUniqueNames)
    
    # from itertools import groupby
    # sliceCountMax = max([len(list(group)) for key, group in groupby(patientValidNames)])
    
    # Add Patinent Number
    for datum in dataFull:
        datum['patientNo'] = patientUniqueNames.index(getPatientNameFromDataFilename(datum['dataFilename']))
    
    indices = {}
    indices['indices4SlicesFixedBySliceTr'], indices['indices4SlicesFixedBySliceTe']     = trainTestSplit(len(dataFull), 0.8, orgBy = 'slice', splBy = 'slice', random = 'fixed')
    indices['indices4SlicesRandBySliceTr'], indices['indices4SlicesRandBySliceTe']       = trainTestSplit(len(dataFull), 0.8, orgBy = 'slice', splBy = 'slice', random = 'random')
    indices['indices4SlicesMixedByPatientTr'], indices['indices4SlicesMixedByPatientTe'] = trainTestSplit(len(dataFull), 0.8, orgBy = 'slice', splBy = 'slice', random = 'mixed')
    indices['indices4SlicesFixedByPatientTr'], indices['indices4SlicesFixedByPatientTe'] = trainTestSplit(len(dataFull), 0.8, orgBy = 'slice', splBy = 'patient', random = 'fixed', dataIDs = patientValidNames)
    indices['indices4SlicesRandByPatientTr'], indices['indices4SlicesRandByPatientTe']   = trainTestSplit(len(dataFull), 0.8, orgBy = 'slice', splBy = 'patient', random = 'random', dataIDs = patientValidNames)
    indices['indices4PatientsFixedTr'], indices['indices4PatientsFixedTe'] = trainTestSplit(NPatients, 0.8, orgBy = 'patient', splBy = 'patient', random = 'fixed')
    indices['indices4PatientsRandTr'], indices['indices4PatientsRandTe']   = trainTestSplit(NPatients, 0.8, orgBy = 'patient', splBy = 'patient', random = 'random')
    
    ifSliceHaveTOS = np.array([not datum.get('fakeTOS', False) for datum in dataFull])
    ifPatitentHaveTOS = [True] * NPatients
    for datum in dataFull:
        ifPatitentHaveTOS[datum['patientNo']] *= not datum.get('fakeTOS', False)
    ifPatitentHaveTOS = np.array(ifPatitentHaveTOS, dtype=np.bool)
    
    indices['indices4SlicesAlllabeledBySliceTr'], indices['indices4SlicesAlllabeledBySliceTe'] = trainTestSplit(len(dataFull), orgBy = 'slice', splBy = 'slice', random = 'allLabeled', ifDataHaveTOS = ifSliceHaveTOS)
    indices['indices4PatientsAlllabeledTr'], indices['indices4PatientsAlllabeledTe']           = trainTestSplit(NPatients, orgBy = 'patient', splBy = 'patient', random = 'allLabeled', ifDataHaveTOS = ifPatitentHaveTOS)
    
    return indices