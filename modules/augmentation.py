# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:41:06 2020

@author: remus
"""

import numpy as np
def augmentationOLD(inputDataRaw, labelDataRaw, config = {}):
    if 'shiftY' in config.keys():
        #print('Aug: Shifiting...')
        for shiftXIdx, shiftX in enumerate(config.get('shiftX', [0])):
            for shiftYIdx, shiftY in enumerate(config['shiftY']):
                inputDataRawShifted = np.roll(np.roll(inputDataRaw, shiftY, axis=-2), shiftX, axis=-1)
                if config['inputType'] not in ['dispField', 'dispFieldJacoImg', 'strainImg']:
                    if config['outputType'] in ['TOS']:
                        labelDataRawShifted = np.roll(labelDataRaw, -shiftY, axis=-1) + shiftX*17
                else:
                    labelDataRawShifted = labelDataRaw
                
                if shiftXIdx + shiftYIdx == 0:
                    inputData = inputDataRawShifted.copy()
                    labelData = labelDataRawShifted.copy()
                else:                                                            
                    inputData = np.concatenate((inputData, inputDataRawShifted), axis=0)
                    labelData = np.concatenate((labelData, labelDataRawShifted), axis=0)
        
    return inputData, labelData

def augmentation(data: list, config={}):
    #dataOri = data
    NDataOri = len(data)
    if 'shiftY' in config.keys():
        for shiftXIdx, shiftX in enumerate(config.get('shiftX', [0])):
            for shiftYIdx, shiftY in enumerate(config['shiftY']):
                if shiftX == 0 and shiftY == 0:
                    continue
                for datum in data[:NDataOri]:
                    dataShifted = np.roll(np.roll(datum[config['inputType']], shiftY, axis=-2), shiftX, axis=-1)
                    if config['inputType'] not in ['dispField', 'dispFieldJacoImg', 'strainImg']:
                        if config['outputType'] in ['TOS', 'TOSInterpolatedMid']:
                            labelShifted = np.roll(datum[config['outputType']], -shiftY, axis=-1) + shiftX*17
                    # CHANGED FOR ISBI
                    # datumNew = {config['inputType']: dataShifted, config['outputType']: labelShifted, 'isAugmented': True}
                    # datumNew = datum.copy()
                    datumNew = {}
                    for key in datum.keys():
                        if key not in ['inputType', 'outputType', 'pred', 'predRaw']:
                            datumNew[key] = datum[key]
                    datumNew[config['inputType']] = dataShifted
                    datumNew[config['outputType']] = labelShifted
                    datumNew['isAugmented'] = True
                    datumNew['patientNo'] = datum.get('patientNo', None)
                    datumNew['augmentation'] = {'shiftX':shiftX, 'shiftY': shiftY}
                    data.append(datumNew)
                    
                # inputDataRawShifted = np.roll(np.roll(inputDataRaw, shiftY, axis=-2), shiftX, axis=-1)
                # if config['inputType'] not in ['dispField', 'dispFieldJacoImg', 'strainImg']:
                #     if config['outputType'] in ['TOS']:
                #         labelDataRawShifted = np.roll(labelDataRaw, -shiftY, axis=-1) + shiftX*17
                # else:
                #     labelDataRawShifted = labelDataRaw
                
                # if shiftXIdx + shiftYIdx == 0:
                #     inputData = inputDataRawShifted.copy()
                #     labelData = labelDataRawShifted.copy()
                # else:                                                            
                #     inputData = np.concatenate((inputData, inputDataRawShifted), axis=0)
                #     labelData = np.concatenate((labelData, labelDataRawShifted), axis=0)
    return data