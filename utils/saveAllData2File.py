# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:35:16 2020

@author: remus
"""

import numpy as np
from datetime import date
from utils.extractCardiacData import getFilenamesGivenPath, extractDataGivenFilenames

bulk_path = '../../Dataset/CRT_TOS_Data_Jerry/'
dataFilenames, labelFilenames, patientIDs = getFilenamesGivenPath(loadProcessed=True)
dataFull, dataFilenamesValid, labelFilenamesValid = extractDataGivenFilenames(dataFilenames, labelFilenames = None, labelInDataFile = True, configs = None, loadAllTypes = True)
np.save(f'../../Dataset/dataFull-{len(dataFull)}-{date.today()}.npy', dataFull)
# np.save(f'./temp/dataFull-{len(dataFull)}-datafilenames.npy', dataFilenamesValid)
# np.save(f'./temp/dataFull-{len(dataFull)}-labelfilenames.npy', labelFilenamesValid)