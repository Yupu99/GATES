
import os
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

run = 30

# submit jobs
for i in range(run):
    os.system(f"sbatch myjob.sl {i+1}")
    time.sleep(0.5)

