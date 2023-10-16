import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
warnings.resetwarnings()

plt.rcParams["pdf.use14corefonts"] = True
models = ['fcn', 'lstm_fcn', 'resnet']
metrics = ['accuracy', 'precision', 'recall', 'f1', 'tss', 'hss1', 'hss2']
path = "/aug_sf/results_10x_256/"

def load_data(model):
    df = pd.read_csv(path + model + '/res.csv')
    return df

def plot(df, model, metric):
    # Filter the DataFrame to select only the relevant columns
    filtered_df = df[['size', metric]]

    # Create a box plot using Seaborn
    plt.figure(figsize=(5, 4))
    sns.boxplot(x='size', y=metric, data=filtered_df)
    plt.grid(True)

    # Customize the plot
    # plt.title(model + '_' + metric)
    plt.xlabel('Syn/real ratio')
    plt.ylabel(metric.upper())
    plt.tight_layout()
    # Show the plot
    plt.savefig(path + model + "_" + metric + '.pdf', format='pdf', dpi=300)
    plt.show()

for model in models:
    for metric in metrics:
        df = load_data(model)
        plot(df, model, metric)

