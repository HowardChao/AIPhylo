import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from PIL import Image
import random
import math
import matplotlib.pyplot as plt
import dendropy
import csv
from dendropy.calculate import treecompare
import sys
import numpy

def main():
    whole_data = np.load('../../Data_Preprocessing/output/500_Dataset/training_data.npy')
    whole_labels_cub = np.load('../../Data_Preprocessing/output/500_Dataset/training_dists_cub.npy')
    whole_labels_flatten = np.load('../../Data_Preprocessing/output/500_Dataset/training_dists_flatten.npy')
    
    BRANCH_NUM = 100
    SEQUENCE_LEN = 1000
    OUTPUT_DIST_NUM = int((BRANCH_NUM - 1) * BRANCH_NUM / 2)
    
    whole_data_before_shuffle = whole_data.reshape((whole_data.shape[0]*100, 1000))
    plt.imshow(whole_data_before_shuffle, cmap='hot', interpolation='nearest', aspect='auto')
    plt.savefig('Plot_before_shuffle.png')

if __name__ == "__main__":
    main()