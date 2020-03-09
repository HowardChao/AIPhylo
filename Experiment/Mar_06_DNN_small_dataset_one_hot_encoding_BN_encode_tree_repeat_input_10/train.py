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
import pickle
from sklearn.model_selection import train_test_split
import string

sys.path.append("../..")

import Bandelt_Encode_Decode.Bandelt_Encode_func as BN_Encode_func
import Bandelt_Encode_Decode.Bandelt_Decode_func as BN_Decode_func
import Bandelt_Encode_Decode.Bandelt_Node as BN

BRANCH_NUM = 100
SEQUENCE_LEN = 1000

Y_DENOMINATOR = []
for i in range(100):
    Y_DENOMINATOR.append(i+1)
    
loss_list_epoch= []
val_loss_list_epoch= []
RF_avg_distance_epoch = []
RF_list_distance_epoch = []

def custom_loss_mean_squared_error(y_true, y_pred):
    global Y_DENOMINATOR
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.square(y_pred*Y_DENOMINATOR - y_true*Y_DENOMINATOR), axis=-1)

    
def main():
    X_Input_Alignment_Data = np.load('../../ALL_DATA/499_dataset_Input_Output/X_Input_Alignment_Data.npy')
    Y_Output_BN_List_Data = np.load('../../ALL_DATA/499_dataset_Input_Output/Y_Output_BN_List_Data.npy')
    File_Name_List = np.load('../../ALL_DATA/499_dataset_Input_Output/File_Name_List.npy')
    
    with open('../../ALL_DATA/mapping_dic_dic.npy', 'rb') as input:
        mapping_dic_dic = pickle.load(input)
    with open('../../ALL_DATA/mapping_dic_dic_decode.npy', 'rb') as input:
        mapping_dic_dic_decode = pickle.load(input)
        
    print('X_Input_Alignment_Data: ', X_Input_Alignment_Data.shape)
    print('Y_Output_BN_List_Data: ', Y_Output_BN_List_Data.shape)
    print('File_Name_List: ', File_Name_List.shape)
    

        
    Y_Output_BN_List_Data = np.true_divide(Y_Output_BN_List_Data, Y_DENOMINATOR)
    
    X_Input_Alignment_Data_repeat = np.zeros((499*10, 100, 1000, 5))
    Y_Output_BN_List_Data_repeat = np.zeros((499*10, 100))
    File_Name_List_repeat = np.empty((499*10), dtype='S100')
    
    for i in range(X_Input_Alignment_Data.shape[0]):
        for repeat_idx in range(10):
            shuffle_indices_100 = random.sample(range(0, 100), 100)
            shuffle_indices_1000 = random.sample(range(0, 1000), 1000)
            tmp = X_Input_Alignment_Data[i][shuffle_indices_100]
            tmp = tmp[:, shuffle_indices_1000]
            inner_idx = 10*i + repeat_idx
            X_Input_Alignment_Data_repeat[inner_idx] = tmp
            Y_Output_BN_List_Data_repeat[inner_idx] = Y_Output_BN_List_Data[i]
            File_Name_List_repeat[inner_idx] = File_Name_List[i]
            
    X_Input_Alignment_Data = X_Input_Alignment_Data_repeat
    Y_Output_BN_List_Data = Y_Output_BN_List_Data_repeat
    File_Name_List = File_Name_List_repeat
    
    print('X_Input_Alignment_Data: ', X_Input_Alignment_Data.shape)
    print('Y_Output_BN_List_Data: ', Y_Output_BN_List_Data.shape)
    print('File_Name_List: ', File_Name_List.shape)
    
    TRAINING_RATION = 0.85

    samples_count = X_Input_Alignment_Data.shape[0]
    train_size = math.floor(TRAINING_RATION*samples_count)
    shuffle_indices = random.sample(range(0, samples_count), samples_count)

    indices_train = shuffle_indices[0:train_size]
    indices_test = shuffle_indices[train_size:samples_count]

    X_train = X_Input_Alignment_Data[indices_train,:]
    Y_train = Y_Output_BN_List_Data[indices_train]
    File_Name_train = File_Name_List[indices_train]

    X_test = X_Input_Alignment_Data[indices_test,:]
    Y_test = Y_Output_BN_List_Data[indices_test]
    File_Name_test = File_Name_List[indices_test]

    print('X_train.shape : ', X_train.shape)
    print('X_test.shape : ', X_test.shape)
    print('Y_train.shape : ', Y_train.shape)
    print('Y_test.shape : ', Y_test.shape)
    print('File_Name_train.shape : ', File_Name_train.shape)
    print('File_Name_test.shape : ', File_Name_test.shape)
    
    np.save('./output/X_train.npy', X_train)
    np.save('./output/X_test.npy', X_test)
    np.save('./output/Y_train.npy', Y_train)
    np.save('./output/Y_test.npy', Y_test)
    np.save('./output/File_Name_train.npy', File_Name_train)
    np.save('./output/File_Name_test.npy', File_Name_test)
    
    ######################################
    #### Here is the model definition ####
    ######################################
    model = Sequential()
    # Layer 1
    model.add(Conv2D(1024, input_shape=X_Input_Alignment_Data.shape[1:], kernel_size=(100,1), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Layer 2
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Layer 3
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Layer 4
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Layer 5
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Layer 6
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Layer 7
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Layer 8
    model.add(Flatten())
    model.add(Dense(BRANCH_NUM, activation='tanh'))
    
    model.summary()
    
    # Model compile
    model.compile(loss=custom_loss_mean_squared_error, optimizer='adam', metrics=['accuracy'])
    
    

    # Do first time to initial the tree!
    print("******** Start fitting model ********")
#     history = model.fit(X_train, Y_train, validation_split=0.15, batch_size=20, epochs=1, verbose=1)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=20, epochs=1, verbose=1)
    print(history.history['loss'])
    loss_list_epoch.append(history.history['loss'][0])
    val_loss_list_epoch.append(history.history['val_loss'][0])

    print("******** Start calculating RF Distance ********")
    RF_list_distance = []
    for index in range(len(X_train)):
        tips_num = np.count_nonzero(np.sum(np.sum(X_train[index], axis=2), axis=1))
        tips_num_selec = tips_num - 3
    #     print("tips_num: ", tips_num)
        #############################################
        #### Get X_train result BN encoding list ####
        #############################################
        X_train_results = model.predict(X_train[index].reshape((1, 100, 1000, 5)))
        X_train_results_trans = np.around(X_train_results[0]*Y_DENOMINATOR)[0:tips_num_selec]
        X_train_results_final = np.concatenate([[0], X_train_results_trans])

        ######################################
        #### Get Y_train BN encoding list ####
        ######################################
        Y_train_trans = np.around(Y_train[index]*Y_DENOMINATOR)[0:tips_num_selec]
        Y_train_final = np.concatenate([[0], Y_train_trans])

        #######################################################
        #### Decode both 'X_train' & 'Y_train' BN encoding ####
        #######################################################
        X_train_results_decode_tree, X_train_results_decode_newick = BN_Decode_func.Bandelt_Decode(list(map(int, X_train_results_final.tolist())), File_Name_train[index].decode('UTF-8'), mapping_dic_dic_decode)
        Y_train_decode_tree, Y_train_decode_newick = BN_Decode_func.Bandelt_Decode(list(map(int, Y_train_final.tolist())), File_Name_train[index].decode('UTF-8'), mapping_dic_dic_decode)

        ################################################################
        #### Dendropy tree creation both both 'X_train' & 'Y_train' ####
        ################################################################
        taxon_namespace = dendropy.TaxonSet()
        X_train_results_tree = dendropy.Tree.get(data=X_train_results_decode_newick, schema="newick", taxon_set=taxon_namespace)
        Y_train_tree = dendropy.Tree.get(data=Y_train_decode_newick, schema="newick", taxon_set=taxon_namespace)
        RF_distance = treecompare.symmetric_difference(X_train_results_tree, Y_train_tree)
        RF_list_distance.append(RF_distance)
    RF_list_distance_epoch.append(RF_list_distance)
    RF_avg_distance_epoch.append(sum(RF_list_distance) / len(RF_list_distance))
    print("RF_list_distance: ", RF_list_distance)
    print("RF_avg_distance_epoch: ", RF_avg_distance_epoch)
    
    
    
    # (5) Train
    # history = model.fit(X_train, Y_train_flatten, validation_data=(X_test, Y_test_flatten), batch_size=16, epochs=10, verbose=1)
    for i in range(10):
        print("******** Start fitting model ********")
#         history = model.fit(X_train, Y_train, validation_split=0.15, batch_size=20, epochs=20, verbose=1)
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=20, epochs=20, verbose=1)
        print(history.history['loss'])
        loss_list_epoch.append(history.history['loss'][0])
        val_loss_list_epoch.append(history.history['val_loss'][0])
        print("******** Start calculating RF Distance ********")
        RF_list_distance = []
        for index in range(len(X_train)):
            tips_num = np.count_nonzero(np.sum(np.sum(X_train[index], axis=2), axis=1))
            tips_num_selec = tips_num - 3
        #     print("tips_num: ", tips_num)
            #############################################
            #### Get X_train result BN encoding list ####
            #############################################
            X_train_results = model.predict(X_train[index].reshape((1, 100, 1000, 5)))
            X_train_results_trans = np.around(X_train_results[0]*Y_DENOMINATOR)[0:tips_num_selec]
            X_train_results_final = np.concatenate([[0], X_train_results_trans])

            ######################################
            #### Get Y_train BN encoding list ####
            ######################################
            Y_train_trans = np.around(Y_train[index]*Y_DENOMINATOR)[0:tips_num_selec]
            Y_train_final = np.concatenate([[0], Y_train_trans])

            #######################################################
            #### Decode both 'X_train' & 'Y_train' BN encoding ####
            #######################################################
            X_train_results_decode_tree, X_train_results_decode_newick = BN_Decode_func.Bandelt_Decode(list(map(int, X_train_results_final.tolist())), File_Name_train[index].decode('UTF-8'), mapping_dic_dic_decode)
            Y_train_decode_tree, Y_train_decode_newick = BN_Decode_func.Bandelt_Decode(list(map(int, Y_train_final.tolist())), File_Name_train[index].decode('UTF-8'), mapping_dic_dic_decode)

            ################################################################
            #### Dendropy tree creation both both 'X_train' & 'Y_train' ####
            ################################################################
            taxon_namespace = dendropy.TaxonSet()
            X_train_results_tree = dendropy.Tree.get(data=X_train_results_decode_newick, schema="newick", taxon_set=taxon_namespace)
            Y_train_tree = dendropy.Tree.get(data=Y_train_decode_newick, schema="newick", taxon_set=taxon_namespace)
            RF_distance = treecompare.symmetric_difference(X_train_results_tree, Y_train_tree)
            RF_list_distance.append(RF_distance)
        RF_list_distance_epoch.append(RF_list_distance)
        RF_avg_distance_epoch.append(sum(RF_list_distance) / len(RF_list_distance))
        print("RF_list_distance: ", RF_list_distance)
        print("RF_avg_distance_epoch: ", RF_avg_distance_epoch)
        
        
    model.save('./output/CNN_model_03.h5')
    np.save('./output/loss_list_epoch.npy', loss_list_epoch)
    np.save('./output/val_loss_list_epoch.npy', val_loss_list_epoch)
    np.save('./output/RF_avg_distance_epoch.npy', RF_avg_distance_epoch)
    np.save('./output/RF_list_distance_epoch.npy', RF_list_distance_epoch)

if __name__ == '__main__':
    main()

