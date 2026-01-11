'''
    script for SSH
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, interpolate


def process_log_data(which_log):
    if not os.path.exists(which_log):
        raise FileNotFoundError(f"The specified path does not exist: {which_log}")

    log_folders = [f.path for f in os.scandir(which_log) if f.is_dir()]

    all_results_df = pd.DataFrame()

    for i, log_path in enumerate(log_folders):
        results_path = f'{log_path}/train_performance/training_record.csv'

        if not os.path.exists(results_path):
            continue

        df = pd.read_csv(results_path, header=0)

        if len(df) < 3:
            continue

        second_column = df.iloc[:, 1]

        all_results_df[f'{i + 1}'] = second_column

    if all_results_df.empty:
        raise ValueError("No valid data found. Please check your log files.")

    return all_results_df


def calculate_confidence_intervals(all_results_df):
    mean_values = all_results_df.mean(axis=1)
    std_error = all_results_df.sem(axis=1)

    confidence_interval = std_error * stats.t.ppf((1 + 0.95) / 2., len(all_results_df.columns) - 1)

    iterations = np.arange(1, len(mean_values) + 1)
    interp_iterations = np.linspace(1, len(mean_values), 500)
    mean_interp = interpolate.interp1d(iterations, mean_values, kind='cubic')
    lower_interp = interpolate.interp1d(iterations, mean_values - confidence_interval, kind='cubic')
    upper_interp = interpolate.interp1d(iterations, mean_values + confidence_interval, kind='cubic')

    return interp_iterations, mean_interp, lower_interp, upper_interp

# Define the log paths
logs = {
    'ES-RL': './...',
    'SPN-CWS': './...',
    'GAT': './...'
}

plt.figure(figsize=(12, 8))

plt.rcParams.update({'font.size': 22})

for algorithm_name, log_path in logs.items():
    all_results_df = process_log_data(log_path)
    interp_iterations, mean_interp, lower_interp, upper_interp = calculate_confidence_intervals(all_results_df)

    plt.plot(interp_iterations, mean_interp(interp_iterations), label=f'{algorithm_name} Convergence')
    plt.fill_between(interp_iterations, lower_interp(interp_iterations), upper_interp(interp_iterations), alpha=0.5, label=f'{algorithm_name} 95% Confidence Interval')

plt.yscale('symlog')
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.ScalarFormatter())
ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
ax.set_yticks([-1000, -600, -300, -100, -50], minor=True)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.xlabel('Generation')
plt.ylabel('-(Total cost)')
plt.title('Convergence on training')
plt.legend()
plt.grid(True)

# Save the figure
output_dir = '../'  # Replace this with the actual save path
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, 'convergence_plot_training.pdf')
plt.savefig(output_file)

plt.show()
