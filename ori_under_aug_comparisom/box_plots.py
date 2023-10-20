

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
warnings.resetwarnings()

plt.rcParams["pdf.use14corefonts"] = True

model = 'resnet'
path = '/Users/peiyuli/PycharmProjects/aug_sf/results_64/'

df = pd.read_csv(path+ model +"/res.csv")

# df.columns

metrics = ['accuracy', 'precision', 'recall', 'f1', 'tss', 'hss1', 'hss2']
aug_methods = df['archive_name'].unique()
# aug_methods

ori = df[df['archive_name'] == aug_methods[0]]
under = df[df['archive_name'] == aug_methods[1]]

def plot(model, metric):
  ww = df[df['archive_name'] == aug_methods[2]][metric]
  ws = df[df['archive_name'] == aug_methods[3]][metric]
  wdba = df[df['archive_name'] == aug_methods[4]][metric]
  tw = df[df['archive_name'] == aug_methods[5]][metric]
  spawner = df[df['archive_name'] == aug_methods[6]][metric]
  rgw = df[df['archive_name'] == aug_methods[7]][metric]
  mw = df[df['archive_name'] == aug_methods[8]][metric]
  dgw = df[df['archive_name'] == aug_methods[9]][metric]

  data = [ww, ws, wdba, tw, spawner, rgw, mw, dgw]
  labels = ['WW', 'WS', 'WDBA', 'TW', 'SPAWNER', 'RGW', 'MW', 'DGW']


  sorted_data, sorted_labels = zip(*sorted(zip(data, labels), key=lambda x: np.mean(x[0])))

  fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

# Rectangular box plot with sorted data and labels
  bplot1 = ax1.boxplot(sorted_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=sorted_labels)  # will be used to label x-ticks

# Tilt x-labels at a specified angle (e.g., 45 degrees)
  ax1.set_xticklabels(sorted_labels, rotation=45)

# Fill with colors
  colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'tan', 'salmon', 'plum']

# Set facecolor of the boxes
  for patch, color in zip(bplot1['boxes'], colors):
      patch.set_facecolor(color)

# Adding horizontal grid lines
  ax1.yaxis.grid(True)
# ax1.set_xlabel('Augmentation methods')
  ax1.set_ylabel(metric)

  plt.tight_layout()
  plt.savefig(path + model + '_'+ metric + ' comparison.pdf', format='pdf', dpi=300)
  plt.show()

for metric in metrics:
  plot(model, metric)

def plot_group(model):
  # Combine 'ori' and 'under' DataFrames into a single DataFrame with a 'Category' column
  ori['Category'] = 'Imbalanced'
  under['Category'] = 'Undersampled'
  combined_data = pd.concat([ori, under], ignore_index=True)

# Transpose the data to have metrics as columns
  combined_data = combined_data.melt(id_vars=['classifier_name', 'Category'], value_vars=metrics,
                                   var_name='Metrics', value_name='Values')

# Create the box plot
  plt.figure(figsize=(5, 4))
  sns.boxplot(data=combined_data, x='Metrics', y='Values', hue='Category', palette='Set2')
  plt.grid(True)
  plt.xlabel('Metrics')
  plt.ylabel('Values')
  # plt.title('Comparison of Metrics Between "ori" and "under" Categories')
  # plt.legend(title='Category')

  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.savefig(path + model + ' comparison.pdf', format='pdf', dpi=300)
  plt.show()

plot_group(model)

"""# fcn"""

model = 'fcn'

df = pd.read_csv(path+ model +"/res.csv")

ori = df[df['archive_name'] == aug_methods[0]]
under = df[df['archive_name'] == aug_methods[1]]

for metric in metrics:
  plot(model, metric)

plot_group(model)

"""# lstm-fcn"""

model = 'lstm_fcn'

df = pd.read_csv(path+ model +"/res.csv")

ori = df[df['archive_name'] == aug_methods[0]]
under = df[df['archive_name'] == aug_methods[1]]

for metric in metrics:
  plot(model, metric)

plot_group(model)

