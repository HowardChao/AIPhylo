import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
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
    training_data = np.load('/floyd/home/DATA/aiphylo/phylo_training_data.npy')
    training_labels = np.load('/floyd/home/DATA/aiphylo/phylo_training_dists_500_5050.npy')
    
    BRANCH_NUM = 100
    SEQUENCE_LEN = 1000
    OUTPUT_DIST_NUM = int((BRANCH_NUM - 1) * BRANCH_NUM / 2)


    samples_count = training_data.shape[0]

    train_size = math.floor(0.85*training_data.shape[0])

    shuffle_indices = random.sample(range(0, samples_count), samples_count)

    indices_train = shuffle_indices[0:train_size]
    indices_test = shuffle_indices[train_size:samples_count]

    X_train = training_data[indices_train,:]
    Y_train = training_labels[indices_train]

    X_test = training_data[indices_test,:]
    Y_test = training_labels[indices_test]

    print('X_train.shape : ', X_train.shape)
    print('X_test.shape : ', X_test.shape)

    print('Y_train.shape : ', Y_train.shape)
    print('Y_test.shape : ', Y_test.shape)
    
    model = Sequential()

    # Layer 1
    model.add(Conv2D(20, input_shape=X_train.shape[1:], kernel_size=(100,1), strides=(1,1)))
    #model.add(Conv2D(48, input_shape=(400,11,1), kernel_size=(2,2), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(1, 1)))

    # Layer 2
    model.add(Conv2D(10, (1,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(1, 4)))

    # Layer 3
    model.add(Conv2D(5, (1,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(1, 4)))

    # Layer 4
    model.add(Conv2D(1, (1,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(1, 2)))

    # Layer 5
    model.add(Flatten())
    model.add(Dense(OUTPUT_DIST_NUM, activation='relu'))

    print(model.summary())

    # Compile 
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=20, epochs=1, verbose=1)
    model.save('./output/CNN_model.h5')

    train_result = model.evaluate(X_train, Y_train)
    test_result = model.evaluate(X_test, Y_test)
    print("Train Acc: ", train_result)
    print("Test Acc: ", test_result)


if __name__ == '__main__':
    main()