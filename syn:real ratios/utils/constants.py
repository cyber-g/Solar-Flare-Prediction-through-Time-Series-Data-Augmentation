ITERATIONS = 5 # nb of random runs for random initializations


DATASET_NAMES = ['solarflare']

sizes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

CLASSIFIERS = ['fcn', 'resnet', 'fcn']
ARCHIVE_NAMES = ['SF_aug_ori', 'SF_aug_tw', 'SF_aug_ws']
DATA_FOLDERS = {'time_warp', 'window_slice'}
data_folders_for_archive = {'SF_aug_ws': 'window_slice', 'SF_aug_tw': 'time_warp'}
