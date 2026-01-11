
"""
    Create a fixed matrix to generate a fixed test_set and fix the arrival time of each workflow
"""
import random
import numpy as np
import pandas as pd
import torch
import yaml
import env.workflow_scheduling_v3.lib.dataset as dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class testSet_Generate():
    def __init__(self, yaml_path):
        self.testMatrix = None
        self.testArrivalTime = None
        self.test_dynamic_gamma = None
        self.test_dynamic_gamma_wf = None

        set_seed(42)  # fix the testing set

        with open(yaml_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            wf_types = 4
            """----test_set------"""
            wf_num_testing = config['env']['wf_num_test']
            validNum = config['env']['validNum']
            self.testMatrix = np.random.randint(0, wf_types, (1, validNum, wf_num_testing))

            # Generate Poisson time matrix
            test_shape = self.testMatrix.shape
            rate = 0.01
            self.testArrivalTime = np.zeros(test_shape)
            for batch in range(test_shape[0]):
                for row in range(test_shape[1]):
                    time_points = [0]
                    for _ in range(test_shape[2] - 1):
                        time_points.append(time_points[-1] + np.random.exponential(1 / rate))
                    self.testArrivalTime[batch, row, :] = time_points

            gamma_all = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 3.0]
            self.test_dynamic_gamma = np.random.choice(gamma_all, size=(test_shape[0], test_shape[1]), replace=True)
            self.test_dynamic_gamma_wf = np.random.choice(gamma_all, size=(test_shape[0], test_shape[1], test_shape[2]), replace=True)

            f.close()



