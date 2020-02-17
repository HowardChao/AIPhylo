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
    plt.savefig('./fig/Plot_before_shuffle.png')
    
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
    plt.savefig('./fig/Plot_after_shuffle.png')
    
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
    plt.savefig('./fig/Plot_shuffle_samples_X_train.png')

    ######################################
    #### Plot after shuffle (X_train) ####
    ######################################
    X_test_shuffle = X_test.reshape((X_test.shape[0]*100, 1000))
    plt.imshow(X_test_shuffle, cmap='hot', interpolation='nearest', aspect='auto')
    plt.savefig('./fig/Plot_shuffle_samples_X_test.png')
    
    np.save('./output/X_train.npy', X_train)
    np.save('./output/Y_train_flatten.npy', Y_train_flatten)
    np.save('./output/X_test.npy', X_test)
    np.save('./output/Y_test_flatten.npy', Y_test_flatten)
    
    
    
    ###############################
    #### Start training models ####
    ###############################
    model = Sequential()
    # Layer 1
    model.add(Conv2D(100, input_shape=X_train.shape[1:], kernel_size=(100,1), strides=(1,1)))
    #model.add(Conv2D(48, input_shape=(400,11,1), kernel_size=(2,2), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Layer 2
    model.add(Conv2D(10, (1, 2), padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Layer 3
    model.add(Conv2D(5, (1, 2), padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Layer 4
    model.add(Conv2D(1, (1, 2), padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Layer 5
    model.add(Flatten())
    model.add(Dense(OUTPUT_DIST_NUM, activation='relu'))
    
    model_summary = model.summary()
    f = open("model_summary.txt", "a")
    f.write(model_summary)
    f.close()
    
    # (4) Compile 
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
    
    loss_list_epoch= []
    RF_avg_distance_epoch = []
    RF_list_distance_epoch = []
    
    ## This function is defined for vector -> symmetric distance matrix
    def utri2mat(utri):
        n = (int(-1 + np.sqrt(1 + 8*len(utri))) // 2) + 1
        iu1 = np.triu_indices(n-1)
        iu1 = (iu1[0], iu1[1] + 1)
        ret = np.zeros((n, n))
        ret[iu1] = utri
        ret.T[iu1] = utri
        return ret

    # Train first time to initial the tree!
    print("******** Start fitting model ********")
    history = model.fit(X_train, Y_train_flatten, validation_data=(X_test, Y_test_flatten), batch_size=50, epochs=1, verbose=1)
    print(history.history['loss'])
    loss_list_epoch.append(history.history['loss'][0])

    print("******** Start calculating RF Distance ********")
    RF_list_distance = []
    for index in range(len(X_train)):
        tips_num = np.count_nonzero(np.sum(X_train[index], axis=1) != 0) 
    #         print("tips_num: ", tips_num)
        array_selection_length = int((tips_num + 1) * tips_num / 2)
        Y_train_flatten_vec_sel = Y_train_flatten[index][0:array_selection_length]
        X_train_results = model.predict(X_train[index].reshape((1, 100, 1000, 1)))
        X_train_results_vec_sel = X_train_results[0][0:array_selection_length]
        original_dis_matrix = utri2mat(Y_train_flatten_vec_sel)
        new_dis_matrix = utri2mat(X_train_results_vec_sel)
        for i in range(len(new_dis_matrix)):
            new_dis_matrix[i,i] = 0
        with open('./output/CSV/original_dis_matrix'+str(index)+'.csv', mode='w') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(original_dis_matrix)
        with open('./output/CSV/new_dis_matrix'+str(index)+'.csv', mode='w') as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(new_dis_matrix)
        taxon_namespace = dendropy.TaxonSet()
        pdm_origin = dendropy.PhylogeneticDistanceMatrix.from_csv(
                src=open('./output/CSV/original_dis_matrix'+str(index)+'.csv'),
                is_first_row_column_names=False,
                is_first_column_row_names=False,
                delimiter=",",
                taxon_namespace = taxon_namespace)
        pdm_new = dendropy.PhylogeneticDistanceMatrix.from_csv(
                src=open('./output/CSV/new_dis_matrix'+str(index)+'.csv'),
                is_first_row_column_names=False,
                is_first_column_row_names=False,
                delimiter=",",
                taxon_namespace = taxon_namespace)
        tree_origin = pdm_origin.nj_tree()
        tree_new = pdm_new.nj_tree()
        RF_distance = treecompare.symmetric_difference(tree_origin, tree_new)
        RF_list_distance.append(RF_distance)
    RF_list_distance_epoch.append(RF_list_distance)
    RF_avg_distance_epoch.append(sum(RF_list_distance) / len(RF_list_distance))
    print("RF_list_distance: ", RF_list_distance)
    print("RF_avg_distance_epoch: ", RF_avg_distance_epoch)
    
    
    # (5) Train
    for i in range(10):
        print("******** Start fitting model ********")
        history = model.fit(X_train, Y_train_flatten, validation_data=(X_test, Y_test_flatten), batch_size=2, epochs=50, verbose=1)
        print(history.history['loss'])
        loss_list_epoch.append(history.history['loss'][0])

        print("******** Start calculating RF Distance ********")
        RF_list_distance = []
        for index in range(len(X_train)):
            tips_num = np.count_nonzero(np.sum(X_train[index], axis=1) != 0) 
            array_selection_length = int((tips_num + 1) * tips_num / 2)
            Y_train_flatten_vec_sel = Y_train_flatten[index][0:array_selection_length]
            X_train_results = model.predict(X_train[index].reshape((1, 100, 1000, 1)))
            X_train_results_vec_sel = X_train_results[0][0:array_selection_length]
            original_dis_matrix = utri2mat(Y_train_flatten_vec_sel)
            new_dis_matrix = utri2mat(X_train_results_vec_sel)
            for i in range(len(new_dis_matrix)):
                new_dis_matrix[i,i] = 0
            with open('./output/CSV/original_dis_matrix'+str(index)+'.csv', mode='w') as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                csvWriter.writerows(original_dis_matrix)
            with open('./output/CSV/new_dis_matrix'+str(index)+'.csv', mode='w') as my_csv:
                csvWriter = csv.writer(my_csv,delimiter=',')
                csvWriter.writerows(new_dis_matrix)
            taxon_namespace = dendropy.TaxonSet()
            pdm_origin = dendropy.PhylogeneticDistanceMatrix.from_csv(
                    src=open('./output/CSV/original_dis_matrix'+str(index)+'.csv'),
                    is_first_row_column_names=False,
                    is_first_column_row_names=False,
                    delimiter=",",
                    taxon_namespace = taxon_namespace)
            pdm_new = dendropy.PhylogeneticDistanceMatrix.from_csv(
                    src=open('./output/CSV/new_dis_matrix'+str(index)+'.csv'),
                    is_first_row_column_names=False,
                    is_first_column_row_names=False,
                    delimiter=",",
                    taxon_namespace = taxon_namespace)
            tree_origin = pdm_origin.nj_tree()
            tree_new = pdm_new.nj_tree()
            RF_distance = treecompare.symmetric_difference(tree_origin, tree_new)
            RF_list_distance.append(RF_distance)
        RF_list_distance_epoch.append(RF_list_distance)
        RF_avg_distance_epoch.append(sum(RF_list_distance) / len(RF_list_distance))
        print("RF_list_distance: ", RF_list_distance)
        print("RF_avg_distance_epoch: ", RF_avg_distance_epoch)

    # Save important variables
    model.save('./output/CNN_model.h5')
    np.save('./output/loss_list_epoch.npy', loss_list_epoch)
    np.save('./output/RF_avg_distance_epoch.npy', RF_avg_distance_epoch)
    np.save('./output/RF_list_distance_epoch.npy', RF_list_distance_epoch)

    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    