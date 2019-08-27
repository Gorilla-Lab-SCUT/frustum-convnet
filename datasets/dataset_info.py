import numpy as np

class KITTICategory(object):
   
    CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    CLASS_MEAN_SIZE = {
        'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
        'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
        'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
    }
 
    NUM_SIZE_CLUSTER = len(CLASSES)

    MEAN_SIZE_ARRAY = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
    for i in range(NUM_SIZE_CLUSTER):
        MEAN_SIZE_ARRAY[i, :] = CLASS_MEAN_SIZE[CLASSES[i]]