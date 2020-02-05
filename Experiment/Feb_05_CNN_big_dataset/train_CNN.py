import numpy as np
import keras
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
np.set_printoptions(threshold=sys.maxsize)

def main():
    BRANCH_NUM = 100
    SEQUENCE_LEN = 1000
    OUTPUT_DIST_NUM = int((BRANCH_NUM - 1) * BRANCH_NUM / 2)
    
    X_train = np.load('./output/X_train.npy')
    Y_train_cub = np.load('./output/Y_train_cub.npy')
    Y_train_flatten = np.load('./output/Y_train_flatten.npy')

    X_test = np.load('./output/X_test.npy')
    Y_test_cub = np.load('./output/Y_test_cub.npy')
    Y_test_flatten = np.load('./output/Y_test_flatten.npy')

    print('X_train.shape : ', X_train.shape)
    print('X_test.shape : ', Y_train_cub.shape)

    print('Y_train_cub.shape : ', Y_train_cub.shape)
    print('Y_test_cub.shape : ', Y_test_cub.shape)
    print('Y_train_flatten.shape : ', Y_train_flatten.shape)
    print('Y_test_flatten.shape : ', Y_test_flatten.shape)
    
    
    model = Sequential()

    # Layer 1
    model.add(Conv2D(10, input_shape=X_train.shape[1:], kernel_size=(3,3), strides=(1,1)))
    #model.add(Conv2D(48, input_shape=(400,11,1), kernel_size=(2,2), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(5, (3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(Conv2D(5, (3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    model.add(Conv2D(1, (3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 5
    model.add(Flatten())
    model.add(Dense(OUTPUT_DIST_NUM, activation='relu'))

    print(model.summary())

    # Compile 
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, Y_train_flatten, validation_data=(X_test, Y_test_flatten), batch_size=20, epochs=200, verbose=1)
    model.save('./output/CNN_model_03.h5')

    train_result = model.evaluate(X_train, Y_train)
    test_result = model.evaluate(X_test, Y_test)
    print("Train Acc: ", train_result)
    print("Test Acc: ", test_result)


if __name__ == '__main__':
    main()