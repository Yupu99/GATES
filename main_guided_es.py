import sys
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

def main():
    yaml_path = 'config/workflow_scheduling_guided_es.yaml'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Let's use: {}".format(device))
    else:
        print("Let's use: cpu")

    # Let BaseConfig parse the config file as usual
    sys.argv = ['main_guided_es.py', '--config', yaml_path]
    baseconfig = BaseConfig()

    cfg = baseconfig.config["yaml-config"]
    env_cfg = cfg["env"]
    opt_cfg = cfg["optim"]

    # Seed override
    yaml_seed = env_cfg.get("seed", 42)
    env_seed = os.getenv("GATES_SEED")

    if env_seed is not None:
        try:
            seed = int(env_seed)
            env_cfg["seed"] = seed
        except ValueError:
            seed = yaml_seed
            env_cfg["seed"] = yaml_seed
    else:
        seed = yaml_seed

    # Guided ES hyperparam overrides (alpha, k, gens, surrogate)
    alpha_env = os.getenv("GATES_ALPHA")
    if alpha_env is not None:
        try:
            opt_cfg["guided_alpha"] = float(alpha_env)
        except ValueError:
            pass  # keep YAML value

    k_env = os.getenv("GATES_SUBSPACE_K")
    if k_env is not None:
        try:
            opt_cfg["subspace_k"] = int(k_env)
        except ValueError:
            pass

    gen_env = os.getenv("GATES_GENERATION_NUM")
    if gen_env is not None:
        try:
            opt_cfg["generation_num"] = int(gen_env)
        except ValueError:
            pass

    # surrogate_episodes should also exist in your YAML (with default, e.g. 2)
    surr_env = os.getenv("GATES_SURR_EPISODES")
    if surr_env is not None:
        try:
            opt_cfg["surrogate_episodes"] = int(surr_env)
        except ValueError:
            pass

    # -------- runtime overrides via env vars --------
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

    # seed-tagged run id for logging
    
    gens = opt_cfg.get("generation_num", "NA")
    os.environ["RUN_TAG"] = f"guided_seed_{seed}_gens_{gens}"
    print("[RUN_TAG]", os.environ["RUN_TAG"], flush=True)

    # Set RNGs
    set_seed(seed)

    # Generate Data
    from config.train_set_config import trainSet_Generate
    from config.test_set_config import testSet_Generate

    train_Set_setting = trainSet_Generate(yaml_path)
    test_Set_setting = testSet_Generate(yaml_path)

    # Train
    Builder(device, baseconfig, train_Set_setting, test_Set_setting).build().train()

if __name__ == "__main__":
    main()
