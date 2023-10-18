ITERATIONS = 5 # nb of random runs for random initializations

# CLASSIFIERS = ['fcn', 'resnet', 'fcn']
# ARCHIVE_NAMES = ['SF_aug_ori', 'SF_aug_under', 'SF_aug_ww', 'SF_aug_ws', 'SF_aug_wdba', 'SF_aug_tw', 'SF_aug_spawner', 'SF_aug_rgw', 'SF_aug_mw', 'SF_aug_dgw']
# DATA_FOLDERS = {'discriminative_guided_warp', 'random_guided_warp', 'time_warp',
#                'window_slice', 'magnitude_warp', 'SPAWNER', 'wdba', 'window_warp', 'original', 'under_sampling'}
#
# data_folders_for_archive = {'SF_aug_ww': 'window_warp', 'SF_aug_wdba': 'wdba', 'SF_aug_spawner': 'SPAWNER',
# 'SF_aug_ws': 'window_slice', 'SF_aug_rgw': 'random_guided_warp', 'SF_aug_mw': 'magnitude_warp',
# 'SF_aug_tw': 'time_warp', 'SF_aug_dgw': 'discriminative_guided_warp',
# 'SF_aug_ori': 'original', 'SF_aug_under': 'under_sampling'}

DATASET_NAMES = ['solarflare']


CLASSIFIERS = ['fcn', 'resnet', 'fcn']
ARCHIVE_NAMES = ['SF_aug_tw', 'SF_aug_ws']
DATA_FOLDERS = {'time_warp', 'window_slice'}
data_folders_for_archive = {'SF_aug_ws': 'window_slice', 'SF_aug_tw': 'time_warp'}
