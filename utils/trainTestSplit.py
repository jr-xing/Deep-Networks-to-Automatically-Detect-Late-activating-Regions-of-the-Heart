import numpy as np
def move_patient_data_fromTrain_toTest(train_indices, test_indices, patientValidNames, patient_names = []):
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