import numpy as np
import pandas as pd
import os
import random
from sklearn.preprocessing import LabelEncoder
import syn_sf_generate.utils.augmentation as aug

random.seed(42)
np.random.seed(42)

results_path = "synthetic_SF/"
for name in ['Solarflare']:
    try:
        print('Dataset:', name)

        print('wdba')
        X_train = pd.read_pickle("data/X_inputs1.pck")
        X_train = np.array(X_train)
        y_train = [0] * 385
        y_train = np.array(y_train)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

        X_aug = aug.wdba(X_train, y_train, batch_size=6, slope_constraint="symmetric",
                         use_window=True)

        if not os.path.exists(os.path.join(results_path, 'wdba', name)):
            os.makedirs(os.path.join(results_path, 'wdba', name), exist_ok=True)

        np.save(os.path.join(results_path, 'wdba', name, 'X_train_aug.npy'), \
                X_aug)

        print('random_guided_warp')
        X_aug = aug.random_guided_warp(X_train, y_train, slope_constraint="symmetric", use_window=True,
                                       dtw_type="normal")
        if not os.path.exists(os.path.join(results_path, 'random_guided_warp', name)):
            os.makedirs(os.path.join(results_path, 'random_guided_warp', name), exist_ok=True)

        np.save(os.path.join(results_path, 'random_guided_warp', name, 'X_train_aug.npy'), \
                X_aug)

        print('discriminative_guided_warp')
        X_aug = aug.discriminative_guided_warp(X_train, y_train, batch_size=6, slope_constraint="symmetric",
                                               use_window=True, dtw_type="normal", use_variable_slice=True)

        if not os.path.exists(os.path.join(results_path, 'discriminative_guided_warp', name)):
            os.makedirs(os.path.join(results_path, 'discriminative_guided_warp', name), exist_ok=True)

        np.save(os.path.join(results_path, 'discriminative_guided_warp', name, 'X_train_aug.npy'), \
                X_aug)

        print('SPAWNER')
        X_aug = aug.spawner(X_train,y_train, sigma=0.05, verbose=0)

        if not os.path.exists(os.path.join(results_path, 'SPAWNER', name)):
            os.makedirs(os.path.join(results_path, 'SPAWNER', name), exist_ok=True)

        np.save(os.path.join(results_path, 'SPAWNER', name, 'X_train_aug.npy'), \
                X_aug)

        print('time_warp')
        X_aug = aug.time_warp(X_train, sigma=0.2, knot=4)
        if not os.path.exists(os.path.join(results_path, 'time_warp')):
            os.makedirs(os.path.join(results_path, 'time_warp', name), exist_ok=True)

        np.save(os.path.join(results_path, 'time_warp', name, 'X_train_aug.npy'), \
                    X_aug)

        print('window_slice')
        X_aug = aug.window_slice(X_train, reduce_ratio=0.9)
        if not os.path.exists(os.path.join(results_path, 'window_slice', name)):
            os.makedirs(os.path.join(results_path, 'window_slice', name), exist_ok=True)

        np.save(os.path.join(results_path, 'window_slice', name, 'X_train_aug.npy'), \
                    X_aug)

        print('window_warp')
        X_aug = aug.window_warp(X_train, window_ratio=0.1, scales=[0.5, 2.])
        if not os.path.exists(os.path.join(results_path, 'window_warp', name)):
            os.makedirs(os.path.join(results_path, 'window_warp', name), exist_ok=True)

        np.save(os.path.join(results_path, 'window_warp', name, 'X_train_aug.npy'), \
                X_aug)

        print('magnitude_warp')
        X_aug = aug.magnitude_warp(X_train, sigma=0.2, knot=4)
        if not os.path.exists(os.path.join(results_path, 'magnitude_warp', name)):
            os.makedirs(os.path.join(results_path, 'magnitude_warp', name), exist_ok=True)

        np.save(os.path.join(results_path, 'magnitude_warp', name, 'X_train_aug.npy'), \
                X_aug)

    except Exception as ex:
        print(ex)










