import matplotlib.pyplot as plt
import pandas as pd

models = ['fcn', 'resnet', 'lstm_fcn']
path = '/Users/peiyuli/PycharmProjects/aug_sf/results_syn_real_binary/'

def plot(model):
    df = pd.read_csv(path + model + "/res.csv")

    # Extract the classifier name and metric names
    classifier_name = df['classifier_name'].iloc[0]
    metric_names = ['accuracy', 'recall', 'precision', 'f1']

    # Extract the mean and std values for each metric
    mean_values = [df[f'{metric}_mean'].iloc[0] for metric in metric_names]
    std_values = [df[f'{metric}_std'].iloc[0] for metric in metric_names]

    # Create a bar plot with error bars
    plt.figure(figsize=(5, 4))
    plt.bar(metric_names, mean_values, yerr=std_values, capsize=5, color='blue', alpha=0.8)
    plt.grid(True)
    # Add labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    # plt.title(f'Binary classification real/syn ({classifier_name})')

    # Show the plot
    plt.tight_layout()
    plt.savefig(path + classifier_name + '_real_syn_binary.pdf', format='pdf', dpi=300)
    plt.show()

for model in models:
    plot(model)
