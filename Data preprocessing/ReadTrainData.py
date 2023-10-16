import pickle
import json
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import resample

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)


####fill nans using numpy
def fillNanInFeature(values_np):
    col_mean = np.nanmean(values_np, axis=0)
    inds = np.where(np.isnan(values_np))
    values_np[inds] = np.take(col_mean, inds[1])
    # print(pd.DataFrame(values_np))
    return values_np


def readClassWiseData(op):
    p1 = 'TSC/train_partition1_data.json'
    p2 = 'TSC/train_partition2_data.json'
    p3 = 'TSC/train_partition3_data.json'


    pp = [p1, p2, p3]

    MVTS_inputs = []
    MVTS_labels = []

    X_inputs = []
    M_inputs = []
    C_inputs = []
    B_inputs = []
    Q_inputs = []

    count = 0
    for p in pp:
        with open(p) as infile:
            for line in infile:
                count += 1

                d = json.loads(line)  # each line is a dictionary
                for key, value in d.items():
                    label = value['label']
                    mvts = value['values']
                    values_df = pd.DataFrame.from_dict(mvts)
                    values_np = values_df.to_numpy()
                    if np.any(np.isnan(values_np)):
                        # print(values_np)
                        # print(count)
                        values_np = fillNanInFeature(values_np)
                        # print(np.any(np.isnan(values_np)))
                    if not np.any(np.isnan(values_np)):
                        # input_signal = values_np.transpose()
                        input_signal = values_np
                        MVTS_inputs.append(input_signal)
                        if label == "X":
                            label = 0
                            X_inputs.append(input_signal)
                        elif label == "M":
                            label = 1
                            M_inputs.append(input_signal)
                        elif label == "C":
                            label = 2
                            C_inputs.append(input_signal)
                        elif label == "B":
                            label = 3
                            B_inputs.append(input_signal)
                        elif label == "Q":
                            label = 4
                            Q_inputs.append(input_signal)
                        MVTS_labels.append(label)

    # print(count)
    print(len(MVTS_inputs))
    print(len(MVTS_labels))
    print(len(X_inputs))
    print(len(M_inputs))
    print(len(C_inputs))
    print(len(B_inputs))
    print(len(Q_inputs))

    save(MVTS_labels, output_path + "labels.pck")
    save(MVTS_inputs, output_path + "inputs.pck")
    # ####Extracting samples of size class X from other classes (Sampling data to balance all the classes.)
    '''
    X_sample = X_inputs
    M_sample = resample(M_inputs, n_samples=samples, replace=False, random_state=22)
    C_sample = resample(C_inputs, n_samples=192, replace=False, random_state=22)
    B_sample = resample(B_inputs, n_samples=193, replace=False, random_state=22)
    Q_sample = resample(Q_inputs, n_samples=samples, replace=False, random_state=22)
    print(len(X_sample))
    print(len(M_sample))
    print(len(C_sample))
    print(len(B_sample))
    print(len(Q_sample))

    Input_Sampled = np.concatenate([X_sample, M_sample, C_sample, B_sample, Q_sample])
    Input_Sampled = np.asarray(Input_Sampled, dtype=np.float_)
    # ###Don't change the order
    labels_Sampled = [0]*385 + [1]*samples + [2]*samples + [3]*samples
    labels_Sampled = np.asarray(labels_Sampled)

    # save(X_sample, output_path + "X_inputs1.pck")


    ###MVTS in numpy format ready for ML
    save(Input_Sampled, output_path + "test_inputs.pck")
    # ###Corresponding MVTS Labels in numpy
    save(labels_Sampled, output_path + "test_labels.pck")
    #
    
    '''
#
output_path = "TSC/Ori_data/"
# samples = 385
readClassWiseData(output_path)
