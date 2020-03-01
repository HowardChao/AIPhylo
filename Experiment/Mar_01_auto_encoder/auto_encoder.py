import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
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
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
sys.path.append("../..")
import Bandelt_Encode_Decode.Bandelt_Encode_func as BN_Encode_func
import Bandelt_Encode_Decode.Bandelt_Decode_func as BN_Decode_func
import Bandelt_Encode_Decode.Bandelt_Node as BN

batch_size = 128
epochs = 50
inChannel = 5
BRANCH_NUM = 100
SEQUENCE_LEN = 1000

def main():
    X_Input_Alignment_Data = np.load('../../ALL_DATA/499_dataset_Input_Output/X_Input_Alignment_Data.npy')
    train_X,valid_X,train_ground,valid_ground = train_test_split(X_Input_Alignment_Data,
                                                                 X_Input_Alignment_Data, 
                                                                 test_size=0.2, 
                                                                 random_state=13)
    
    def autoencoder(input_img):
            #encoder
            #input = 28 x 28 x 1 (wide and thin)
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #100 x 1000 x 32
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #50 x 500 x 64
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #25 x 250 x 128 (small and thick)

            #decoder
            conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #25 x 250 x 128
            up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
            conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 50 x 500 x 64
            up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
            decoded = Conv2D(5, (3, 3), activation='sigmoid', padding='same')(up2) # 100 x 1000 x 1
            return decoded
        
    input_img = Input(shape = (BRANCH_NUM, SEQUENCE_LEN, inChannel))
    autoencoder = Model(input_img, autoencoder(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

    autoencoder.summary()
    
    autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))

    autoencoder.save('./output/autoencoder_model.h5')

if __name__ == '__main__':
    main()