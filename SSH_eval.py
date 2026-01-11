
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

which_log = 'logs/WorkflowScheduling-v3'
log_folders = [f.path for f in os.scandir(which_log) if f.is_dir()]

model_num = 2000  # test the result on the specific model
Gamma = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25]
WF_Size = ['S', 'M', 'L']
for gamma in Gamma:
    for wf_size in WF_Size:
        for log_path in log_folders:
            dir_csv = log_path + "/test_performance_final" + "/testing_record_"+str(gamma)+"_"+str(wf_size)+"_"+str(model_num)+".csv"
            if os.path.exists(dir_csv):
                os.remove(dir_csv)
for gamma in Gamma:
    for wf_size in WF_Size:
        for log_path in log_folders:
            os.system(f"sbatch myjob_eval.sl {gamma} {wf_size} {log_path} {model_num}")

