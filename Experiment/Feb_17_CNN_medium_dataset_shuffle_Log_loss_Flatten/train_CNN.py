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
    whole_labels_flatten = np.load('../../Data_Preprocessing/output/500_Dataset/training_dists_flatten.npy')
    
    BRANCH_NUM = 100
    SEQUENCE_LEN = 1000
    OUTPUT_DIST_NUM = int((BRANCH_NUM - 1) * BRANCH_NUM / 2)
    
    ##########################################
    #### Plot before shuffle (whole_data) ####
    ##########################################
    whole_data_before_shuffle = whole_data.reshape((whole_data.shape[0]*100, 1000))
    plt.imshow(whole_data_before_shuffle, cmap='hot', interpolation='nearest', aspect='auto')
    plt.savefig('./output/fig/Plot_before_shuffle.png')
    
    # Shuffle with the same index
    shuffle_indices_100 = random.sample(range(0, 100), 100)
    shuffle_indices_1000 = random.sample(range(0, 1000), 1000)    
    for i in range(whole_data.shape[0]):
        tmp = whole_data[i][shuffle_indices_100]
        tmp = tmp[:, shuffle_indices_1000]
        whole_data[i] = tmp

    #########################################
    #### Plot after shuffle (whole_data) ####
    #########################################
    whole_data_after_shuffle = whole_data.reshape((whole_data.shape[0]*100, 1000))
    plt.imshow(whole_data_after_shuffle, cmap='hot', interpolation='nearest', aspect='auto')
    plt.savefig('./output/fig/Plot_after_shuffle.png')
    
    # Shuffle sample index
    samples_count = whole_data.shape[0]
    train_size = math.floor(0.85*whole_data.shape[0])
    shuffle_indices = random.sample(range(0, samples_count), samples_count)
        
    indices_train = shuffle_indices[0:train_size]
    indices_test = shuffle_indices[train_size:samples_count]
    
    print("######## Training Data ########")
    X_train = whole_data[indices_train,:]
    Y_train_flatten = whole_labels_flatten[indices_train]
    print("######## Validation Data ########")
    X_test = whole_data[indices_test,:]
    Y_test_flatten = whole_labels_flatten[indices_test]

    print('X_train.shape : ', X_train.shape)
    print('X_test.shape : ', X_test.shape)
    print('Y_train_flatten.shape : ', Y_train_flatten.shape)
    print('Y_test_flatten.shape : ', Y_test_flatten.shape)# Training Testing Dataset Partition
    
    
    #######################################
    #### Plot before shuffle (X_train) ####
    #######################################
    X_train_shuffle = X_train.reshape((X_train.shape[0]*100, 1000))
    plt.imshow(X_train_shuffle, cmap='hot', interpolation='nearest', aspect='auto')
    plt.savefig('./output/fig/Plot_before_shuffle_samples.png')

    ######################################
    #### Plot after shuffle (X_train) ####
    ######################################
    X_test_shuffle = X_test.reshape((X_test.shape[0]*100, 1000))
    plt.imshow(X_test_shuffle, cmap='hot', interpolation='nearest', aspect='auto')
    plt.savefig('./output/fig/Plot_after_shuffle_samples.png')
    
    np.save('./output/X_train.npy', X_train)
    np.save('./output/Y_train_flatten.npy', Y_train_flatten)
    np.save('./output/X_test.npy', X_test)
    np.save('./output/Y_test_flatten.npy', Y_test_flatten)



if __name__ == "__main__":
    main()