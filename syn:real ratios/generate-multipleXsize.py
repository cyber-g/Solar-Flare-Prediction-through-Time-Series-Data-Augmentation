# This is to concat the synthetic X class with the original X class into different syn/real ratio x-class samples
import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter

input_path = '/data/'
output_path = '/10times_aug_res/'


inputs = pd.read_pickle(input_path + "Sampled_inputs5.pck")
labels = pd.read_pickle(input_path + "Sampled_labels5.pck")

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile
            return None
        return directory_path

methods = ['window_slice']
ratios = np.arange(start=0.1, stop=1.1, step=0.1, dtype=np.float32)

def generate_synthetic(method, size):
    xlabel = np.asanyarray([0] * 385 * size)
    x_synthetic = np.load('/10times_aug/' + method + '/Solarflare/' + str(ratios[size-1]) + '/X_train_aug.npy')
    # x_train_syn1 = np.load('synthetic_X-class/' + method + '/Solarflare/' + str(sigmas[1]) + '/X_train_aug.npy')

    x_train_syn1 = pd.read_pickle('/10times_aug_res/' + method + '/' + str(size-1) + 'x/augmented_inputs.pck')
    x_synthetic = np.concatenate((x_synthetic, x_train_syn1), axis=0)
    return xlabel, x_synthetic


sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for method in methods:
    for size in sizes:
        xlabel, x_train_syn = generate_synthetic(method, size)
        x_augmented = x_train_syn
        label = np.concatenate((xlabel, labels), axis=0)
        # input = np.concatenate((x_augmented, inputs), axis=0)
        label_count = Counter(label)
        create_directory(output_path + method + '/' + str(size) +'x')
        save(x_augmented, output_path + method + '/' + str(size) +'x' + "/augmented_inputs.pck")
        save(label, output_path + method + '/' + str(size) +'x' + "/augmented_labels.pck")
        print(label_count)
        print(x_augmented.shape, label.shape)


'''
methods = ['time_warp']
sigmas = np.arange(start=0.1, stop=0.31, step=0.025, dtype=np.float32)

def generate_synthetic(method, size):
    xlabel = np.asanyarray([0] * 385 * size)
    x_synthetic = np.load('/Users/peiyuli/PycharmProjects/aug_sf/10times_aug/' + method + '/Solarflare/' + str(sigmas[size-1]) + '/X_train_aug.npy')
    # x_train_syn1 = np.load('/aug_sf/synthetic_X-class/' + method + '/Solarflare/' + str(sigmas[1]) + '/X_train_aug.npy')

    # x_train_syn1 = pd.read_pickle('/aug/magnitude_warp/8x/augmented_inputs.pck')
    # x_synthetic = np.concatenate((x_train_syn0, ori_X), axis=0)
    return xlabel, x_synthetic

# sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9]


for method in methods:
    for size in sizes:
        xlabel, x_train_syn = generate_synthetic(method, size)
        x_augmented = x_train_syn
        label = np.concatenate((xlabel, labels), axis=0)
        input = np.concatenate((x_augmented, inputs), axis=0)
        label_count = Counter(label)
        create_directory(output_path + method + '/' + str(size) +'x')
        save(input, output_path + method + '/' + str(size) +'x' + "/augmented_inputs.pck")
        save(label, output_path + method + '/' + str(size) +'x' + "/augmented_labels.pck")
        print(label_count)
        print(input.shape, label.shape)
        
'''



