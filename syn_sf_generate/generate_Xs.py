import numpy as np
import os
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import syn_sf_generate.utils.augmentation as aug

random.seed(42)
np.random.seed(42)

results_path = "synthetic_SF/"
for name in ['Solarflare']:
    try:
        print('Dataset:', name)
        X_train = pd.read_pickle("data/X_inputs1.pck")
        X_train = np.array(X_train)
        y_train = [0] * 385
        y_train = np.array(y_train)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

        print('time_warp')
        sigmas = np.arange(start=0.1, stop=0.31, step=0.025, dtype=np.float32)
        for sigma in sigmas:
            X_aug = aug.time_warp(X_train, sigma=sigma, knot=4)

            if not os.path.exists(os.path.join(results_path, 'time_warp', name, str(sigma))):
                os.makedirs(os.path.join(results_path, 'time_warp', name, str(sigma)), exist_ok=True)

            np.save(os.path.join(results_path, 'time_warp', name, str(sigma), 'X_train_aug.npy'), \
                    X_aug)
        

        print('window_slice')
        ratios = np.arange(start=0.1, stop=1.1, step=0.1, dtype=np.float32)
        for ratio in ratios:
            X_aug = aug.window_slice(X_train, reduce_ratio=ratio)

            if not os.path.exists(os.path.join(results_path, 'window_slice', name, str(ratio))):
                os.makedirs(os.path.join(results_path, 'window_slice', name, str(ratio)), exist_ok=True)

            np.save(os.path.join(results_path, 'window_slice', name, str(ratio), 'X_train_aug.npy'), \
                    X_aug)


    except Exception as ex:
        print(ex)
    
    


    
    
    
    
    
    
