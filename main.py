
"""
    Main Script for SSH
"""

import argparse
import os
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

    cfg = baseconfig.config["yaml-config"]
    env_cfg = cfg["env"]
    opt_cfg = cfg["optim"]

    # Seed override: default to run-based seed, allow env override
    seed = run * 100 + 1
    env_seed = os.getenv("GATES_SEED")
    if env_seed is not None:
        try:
            seed = int(env_seed)
            env_cfg["seed"] = seed
        except ValueError:
            pass

    # Override generation count (for smoke runs)
    gen_env = os.getenv("GATES_GENERATION_NUM")
    if gen_env is not None:
        try:
            opt_cfg["generation_num"] = int(gen_env)
        except ValueError:
            pass

    # Runtime overrides
    eval_env = os.getenv("GATES_EVAL_EP_NUM")
    if eval_env is not None:
        try:
            baseconfig.config["runtime-config"]["eval_ep_num"] = int(eval_env)
        except ValueError:
            pass

    proc_env = os.getenv("GATES_PROCESSOR_NUM")
    if proc_env is not None:
        try:
            baseconfig.config["runtime-config"]["processor_num"] = int(proc_env)
        except ValueError:
            pass

    # Set global running seed
    set_seed(seed)
    print(f"seed:{seed}")

    from config.train_set_config import trainSet_Generate
    yaml_path = 'config/workflow_scheduling_es_openai.yaml'
    train_Set_setting = trainSet_Generate(yaml_path)

    # Start assembling RL and training process
    Builder(device, baseconfig, train_Set_setting, test_setting).build().train()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Let's use: {}".format(device))
    else:
        print("Let's use: cpu")

    from config.test_set_config import testSet_Generate
    yaml_path = 'config/workflow_scheduling_es_openai.yaml'
    test_Set_setting = testSet_Generate(yaml_path)

    NeSi_parser = argparse.ArgumentParser(description='settings func', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    NeSi_parser.add_argument('--run', '-r', type=int, required=True, help='the run number')
    NeSi_args = NeSi_parser.parse_args()
    print(f"run:{NeSi_args.run}")

    main(device, NeSi_args.run, test_Set_setting)
