
"""
    Main Script for SSH
"""

import argparse
import random
import numpy as np
import torch
from config.base_config import BaseConfig
from builder import Builder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(device, run, test_setting):
    baseconfig = BaseConfig()

    # Set global running seed
    set_seed(run*100+1)
    print(f"seed:{run*100+1}")

    from config.train_set_config import trainSet_Generate
    yaml_path = baseconfig.config['runtime-config']['config']
    train_Set_setting = trainSet_Generate(yaml_path)

    # Start assembling RL and training process
    Builder(device, baseconfig, train_Set_setting, test_setting).build().train()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Let's use: {}".format(device))
    else:
        print("Let's use: cpu")

    # Pre-parse --config so test_Set_setting uses the correct yaml
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument('--config', type=str, default='config/workflow_scheduling_es_openai.yaml')
    _pre_args, _ = _pre_parser.parse_known_args()

    from config.test_set_config import testSet_Generate
    test_Set_setting = testSet_Generate(_pre_args.config)

    NeSi_parser = argparse.ArgumentParser(description='settings func', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    NeSi_parser.add_argument('--run', '-r', type=int, required=True, help='the run number')
    NeSi_args = NeSi_parser.parse_args()
    print(f"run:{NeSi_args.run}")

    main(device, NeSi_args.run, test_Set_setting)
