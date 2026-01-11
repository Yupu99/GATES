import os
import numpy as np
import pandas as pd

logs = {
    'name': 'GATES',
    'path': '.../',
}

Gamma = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25]
WF_Size = ['S', 'M', 'L']

if not os.path.exists(logs['path']):
    raise FileNotFoundError(f"The specified path does not exist: {logs['path']}")

log_folders = [f.path for f in os.scandir(logs['path']) if f.is_dir()]

all_data = []

for gamma in Gamma:
    for wf_size in WF_Size:
        data = []
        for log_path in log_folders:
            results_path = os.path.join(log_path, 'test_performance_final')
            csv_path = os.path.join(results_path, f"testing_record_{gamma}_{wf_size}.csv")

            if not os.path.exists(csv_path):
                continue

            try:
                df = pd.read_csv(csv_path, header=None)
                if df.empty:
                    print(f"Data is empty! {csv_path}")
                    continue

                total_cost = abs(float(df.iloc[0, 1]))
                VM_cost = float(df.iloc[0, -3])
                SLA_Penalty = float(df.iloc[0, -2])
                data.append([total_cost, VM_cost, SLA_Penalty])
            except Exception as e:
                print(f"Failed to read {csv_path}: {e}")
                continue

        if len(data) == 0:
            print(f"No data for gamma={gamma}, wf_size={wf_size}")
            continue

        df_one = pd.DataFrame(data, columns=["total_cost", "VM_cost", "SLA_penalty"])
        mean = df_one.mean()
        std = df_one.std()

        all_data.append([f"{gamma}_{wf_size}", mean["total_cost"], std["total_cost"], mean["VM_cost"], mean["SLA_penalty"]])

columns = ['Scenario', 'total_cost_mean', 'total_cost_std', 'VM_cost_mean', 'SLA_penalty_mean']
df_all = pd.DataFrame(all_data, columns=columns)

output_dir = '.../results_summary'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, f"{logs['name']}_summary.csv")
df_all.to_csv(output_path, index=False)
print(f"\nThe results have been saved to：{output_path}")

