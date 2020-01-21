#!/usr/bin/env python3
import pandas
from itertools import product
import sys, argparse, os
import numpy as np
from math import log, ceil
from scipy.stats import multinomial, chi2
from math import factorial
import re

folder_list = ['Bergsten_2013', 'Broughton_2013', 'Brown_2012', 'Cannon_2016_dna']

dic={'A':'1', 'T':'2', 'C':'3', 'G':'4', '-':'0', '?':'0'}

all_data_list = []
all_dists_list = []

for folder in folder_list:

    filepath = './training/' + folder + '/'
    files_list = os.listdir(filepath)

    for file in files_list:
        if file.find('.nex.treefile.dist') > 0:
            file_base_name = file[0:file.find('.nex.treefile.dist')]
            print(file_base_name)

            seq_data_raw = open(filepath+file_base_name+'.nex')
            seq_dists_raw = open(filepath+file_base_name+'.nex.treefile.dist')

            seq_data = seq_data_raw.readlines()[6:]

            all_seq_data_list = []
            #all_seq_dists_list = []

            all_seq_data = np.zeros((100,1000,1))
            all_seq_dists = np.zeros((100,100,1))

            i = 0

            for line in seq_data:
                curr_line = line.split()
                if len(curr_line) < 2:
                    break

                seq_data_str = curr_line[1]

                print('===============')
                print(i)

                print(seq_data_str)

                for item in dic:
                    seq_data_str = seq_data_str.replace(item, dic[item])

                #replace any remaining letter chars with 0                
                seq_data_str = re.sub(r'[A-Z]', r'0', seq_data_str)

                print((seq_data_str))

                seq_data_list = list(seq_data_str)

                #print('seq_data_array.shape : ', seq_data_array.shape)

                if len(seq_data_list) > 1000:
                    seq_data_list = seq_data_list[0:999]

                all_seq_data[i,0:len(seq_data_list),0] = seq_data_list

                i+=1

            all_data_list.append(all_seq_data)

            print('all_seq_data.shape: ', all_seq_data.shape)

            # NOW PROCESS THE OUTPUT (distances)

            seq_dists = seq_dists_raw.readlines()[1:]

            j = 0
            for line in seq_dists:
                curr_line = line.split()
                if len(curr_line) < 2:
                    break
                j+=1

                dist_values_array = np.array(curr_line[1:])
                dist_values_array = dist_values_array.astype('float')
                print(dist_values_array.shape)

                all_seq_dists[j,0:dist_values_array.shape[0],0] = dist_values_array

            all_dists_list.append(all_seq_dists)

            print('all_seq_dists.shape: ', all_seq_dists.shape)
            print('*******************************')

training_data = np.stack(all_data_list,axis=0)
training_dists = np.stack(all_dists_list,axis=0)

print('training_data.shape: ', training_data.shape)
print('training_dists.shape: ', training_dists.shape)

np.save('training_data.npy', training_data)
np.save('training_dists.npy', training_dists)





