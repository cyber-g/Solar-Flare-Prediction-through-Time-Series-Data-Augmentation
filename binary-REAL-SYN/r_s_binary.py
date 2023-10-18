import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.utils import load_real_syn_binary
from utils.utils import create_directory, generate_results_syn_real_binary
from utils.constants import sizes

import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES, DATASET_NAMES
from utils.constants import ITERATIONS

def fit_classifier(aug_method):
    x_train, x_test, y_train, y_test = load_real_syn_binary(aug_method)
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'fcn':
        from classifiers_syn_real_binary import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers_syn_real_binary import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'lstm_fcn':
        from classifiers_syn_real_binary import mlstm_fcn
        return mlstm_fcn.Classifier_LSTM_FCN(output_directory, input_shape, nb_classes, verbose)
# change this directory for your machine
root_dir = '/SF'

if sys.argv[1] == 'run_all':
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)
        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)
            for iter in range(ITERATIONS):
                print('\t\titer', iter)
                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)
                tmp_output_directory = root_dir + '/results_syn_real_binary/' + classifier_name + '/' + archive_name + trr + '/'

                for dataset_name in DATASET_NAMES:
                    print('\t\t\tdataset_name: ', dataset_name)

                    output_directory = tmp_output_directory + dataset_name + '/'

                    create_directory(output_directory)
                    aug_method = utils.constants.data_folders_for_archive[archive_name]
                    fit_classifier(aug_method)

                    print('\t\t\t\tDONE')

                    # the creation of this directory means
                    create_directory(output_directory + '/DONE')


elif sys.argv[1] == 'generate_results_csv':
    res = generate_results_syn_real_binary('res.csv', root_dir)
    # print(res.to_string())
    # res.to_csv(root_dir + "/results_binary/", index=False)
else:
    # this is the code used to launch an experiment on a dataset
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name = sys.argv[3]
    itr = sys.argv[4]

    if itr == '_itr_0':
        itr = ''

    output_directory = root_dir + '/results_syn_real_binary/' + classifier_name + '/' + archive_name + itr + '/' + \
                       dataset_name + '/'

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)

        fit_classifier()

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')

