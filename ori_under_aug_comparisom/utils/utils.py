import numpy as np
import pandas as pd
import matplotlib

from utils.constants import ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import ITERATIONS as ITERATIONS
from utils.constants import CLASSIFIERS as CLASSIFIERS
from utils.constants import sizes
from sklearn.metrics import confusion_matrix

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*size changed.*", category=RuntimeWarning)

def load_data(method):
    inputs = pd.read_pickle("SF/aug/"+method+"/augmented_inputs.pck")
    labels = pd.read_pickle("SF/aug/"+method+"/augmented_labels.pck")
    inputs = inputs.transpose(0, 2, 1)
    print(inputs.shape, labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, random_state=42)
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    return X_train, X_test, y_train, y_test

def load_data_binary(method):
    inputs = pd.read_pickle("SF/aug/"+method+"/augmented_inputs.pck")
    labels = pd.read_pickle("SF/aug/"+method+"/augmented_labels.pck")
    inputs = inputs.transpose(0, 2, 1)

    labels = np.where(labels == 0, 1, labels)
    labels = np.where(((labels == 2) | (labels == 3)), 0, labels)
    print(labels)

    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, random_state=42)
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std


    return X_train, X_test, y_train, y_test

def load_aug(method, size):
    if size == 0:
        inputs = pd.read_pickle(
            "SF/data/" + "Sampled_inputs5.pck")
        labels = pd.read_pickle(
            "SF/data/" + "Sampled_labels5.pck")
    else:
        inputs = pd.read_pickle("SF/10times_aug_res/" + method + '/' + str(size) +"x/augmented_inputs.pck")
        labels = pd.read_pickle("SF/10times_aug_res/" + method + '/' + str(size) +"x/augmented_labels.pck")

    inputs = inputs.transpose(0, 2, 1)

    labels = np.where(labels == 0, 1, labels)
    labels = np.where(((labels == 2) | (labels == 3)), 0, labels)

    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, random_state=42)
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    return X_train, X_test, y_train, y_test

def load_real_syn_binary(aug_method):

    real_X = pd.read_pickle("SF/data/X_inputs1.pck")
    real_y = [0] * 385

    syn_X = np.load("SF/synthetic_X-class/" + aug_method + "/Solarflare/X_train_aug.npy")
    syn_y = [1] * 385

    real_X = np.array(real_X)
    syn_X = np.array(syn_X)

    inputs = np.concatenate((real_X, syn_X), axis=0)
    labels = real_y + syn_y
    labels = np.array(labels)

    inputs = inputs.transpose(0, 2, 1)

    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, random_state=42)
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    tss = (TP / (TP + FN)) - (FP / (FP + TN))
    hss1 = ((TP + TN) - (FP + TN)) / (TP + FN)
    hss2 = 2 * ((TP * TN) - (FN * FP)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

    res = pd.DataFrame(data=np.zeros((1, 8), dtype=np.float), index=[0],
                       columns=[ 'accuracy', 'precision', 'recall', 'f1', 'tss', 'hss1', 'hss2', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['f1'] = f1_score(y_true, y_pred, average='macro')
    res['tss'] = tss
    res['hss1'] = hss1
    res['hss2'] = hss2
    res['duration'] = duration
    return res

# # for case study on four class classificatioin
# def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
#     TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
#     res = pd.DataFrame(data=np.zeros((1, 8), dtype=np.float), index=[0],
#                        columns=[ 'accuracy', 'precision', 'recall', 'f1', 'duration'])
#     res['precision'] = precision_score(y_true, y_pred, average='macro')
#     res['accuracy'] = accuracy_score(y_true, y_pred)
#     res['recall'] = recall_score(y_true, y_pred, average='macro')
#     res['f1'] = f1_score(y_true, y_pred, average='macro')
#     res['duration'] = duration
#     return res


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

def generate_results_csv(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'dataset_name', 'archive_name', 'accuracy', 'precision', 'recall', 'f1', 'tss',
                     'hss1', 'hss2', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                output_dir = root_dir + '/results_64/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue

                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['dataset_name'] = 'solar_flare'
                df_metrics['archive_name'] = archive_name

                res = pd.concat((res, df_metrics), axis=0, sort=False)
        print(root_dir + "/results_64/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_64/" + classifier_name + "/" + output_file_name, index=False)
    return res

# def generate_results_csv(output_file_name, root_dir):
#     for classifier_name in CLASSIFIERS:
#         res = pd.DataFrame(
#             columns=['classifier_name', 'dataset_name', 'archive_name', 'accuracy', 'precision', 'recall', 'f1', 'duration'])
#         for archive_name in ARCHIVE_NAMES:
#             for it in range(ITERATIONS):
#                 curr_archive_name = archive_name
#                 if it != 0:
#                     curr_archive_name = curr_archive_name + '_itr_' + str(it)
#                 output_dir = root_dir + '/results_four_aug/' + classifier_name + '/' \
#                                  + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
#                 if not os.path.exists(output_dir):
#                     continue
#
#                 df_metrics = pd.read_csv(output_dir)
#                 df_metrics['classifier_name'] = classifier_name
#                 df_metrics['dataset_name'] = 'solar_flare'
#                 df_metrics['archive_name'] = archive_name
#
#                 res = pd.concat((res, df_metrics), axis=0, sort=False)
#         print(root_dir + "/results_four_aug/" + classifier_name + "/" + output_file_name)
#
#         res.to_csv(root_dir + "/results_four_aug/" + classifier_name + "/" + output_file_name, index=False)
#     return res


def generate_results_csv_10x(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'archive_name', 'size', 'accuracy', 'precision', 'recall', 'f1', 'tss',
                     'hss1', 'hss2', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for size in sizes:
                for it in range(ITERATIONS):
                    curr_archive_name = archive_name
                    if it != 0:
                        curr_archive_name = curr_archive_name + '_itr_' + str(it)
                    output_dir = root_dir + '/results_10x_256/' + str(size) + '/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'

                    if not os.path.exists(output_dir):
                        continue

                    df_metrics = pd.read_csv(output_dir)
                    df_metrics['classifier_name'] = classifier_name
                    df_metrics['size'] = size
                    df_metrics['archive_name'] = archive_name

                    res = pd.concat((res, df_metrics), axis=0, sort=False)
        print(root_dir + "/results_10x_256/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_10x_256/" + classifier_name + "/" + output_file_name, index=False)
        return res

def generate_results_syn_real_binary(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'archive_name', 'accuracy', 'precision', 'recall', 'f1', 'tss',
                     'hss1', 'hss2', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                output_dir = root_dir + '/results_syn_real_binary/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue

                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['archive_name'] = archive_name

                res = pd.concat((res, df_metrics), axis=0, sort=False)
        res = pd.DataFrame({
            'accuracy_mean': res.groupby(
                ['classifier_name'])['accuracy'].mean(),
            'accuracy_std': res.groupby(
                ['classifier_name'])['accuracy'].std(),
            'recall_mean': res.groupby(
                ['classifier_name'])['recall'].mean(),
            'recall_std': res.groupby(
                ['classifier_name'])['recall'].std(),
            'precision_mean': res.groupby(
                ['classifier_name'])['precision'].mean(),
            'precision_std': res.groupby(
                ['classifier_name'])['precision'].std(),
            'f1_mean': res.groupby(
                ['classifier_name'])['f1'].mean(),
            'f1_std': res.groupby(
                ['classifier_name'])['f1'].std()
        }).reset_index()

        print(root_dir + "/results_syn_real_binary/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_syn_real_binary/" + classifier_name + "/" + output_file_name, index=False)

    return res

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics

def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)

def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)
    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics

