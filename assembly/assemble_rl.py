import multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import shutil
import time
from collections import deque
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import torch
import pickle as pickle
import yaml
from torch_geometric.utils import from_networkx

from assembly.base_assemble import BaseAssembleRL
from utils.running_mean_std import RunningMeanStd

from utils.policy_dict import agent_policy
import builder


class AssembleRL(BaseAssembleRL):

    def __init__(self, device, config, env, env_test, policy, optim, train_Set_setting, test_Set_setting):
        super(AssembleRL, self).__init__()
        self.device = device
        self.config = config
        self.env = env
        self.env_test = env_test
        self.policy = policy
        self.optim = optim

        # will not be used in GATES
        self.dynamic_gamma = self.config.config['yaml-config']["env"]["dynamic_gamma"]
        self.train_dynamic_gamma = train_Set_setting.train_dynamic_gamma
        self.test_dynamic_gamma = test_Set_setting.test_dynamic_gamma

        # will not be used in GATES
        self.dynamic_gamma_wf = self.config.config['yaml-config']["env"]["dynamic_gamma_wf"]
        self.train_dynamic_gamma_wf = train_Set_setting.train_dynamic_gamma_wf
        self.test_dynamic_gamma_wf = test_Set_setting.test_dynamic_gamma_wf

        self.running_mstd = self.config.config['yaml-config']["optim"]['input_running_mean_std']
        if self.running_mstd:
            self.ob_rms = RunningMeanStd(shape=self.env.observation_space.shape)
            self.ob_rms_mean = self.ob_rms.mean
            self.ob_rms_std = np.sqrt(self.ob_rms.var)
        else:
            self.ob_rms = None
            self.ob_rms_mean = None
            self.ob_rms_std = None

        self.generation_num = self.config.config['yaml-config']['optim']['generation_num']
        self.processor_num = self.config.config['runtime-config']['processor_num']
        self.eval_ep_num = self.config.config['yaml-config']['env']['evalNum']
        self.valid_ep_num = self.config.config['yaml-config']['env']['validNum']

        # log settings
        self.log = self.config.config['runtime-config']['log']
        self.save_model_freq = self.config.config['runtime-config']['save_model_freq']
        self.save_mode_dir = None

        self.train_or_test = None

    def train(self):
        if self.log:
            now = datetime.now()
            curr_time = now.strftime("%Y%m%d%H%M%S%f")
            dir_lst = []
            self.save_mode_dir = f"logs/{self.env.name}/{curr_time}_{os.getpid()}"
            dir_lst.append(self.save_mode_dir)
            dir_lst.append(self.save_mode_dir + "/saved_models/")
            for _dir in dir_lst:
                os.makedirs(_dir, exist_ok=True)

            with open(self.save_mode_dir + "/profile.yaml", 'w') as file:
                yaml.dump(self.config.config['yaml-config'], file)
                file.close()

        self.policy.to(self.device)
        population = self.optim.init_population(self.policy, self.env)

        if self.config.config['yaml-config']['optim']['maximization']:
            best_reward_so_far = float("-inf")
        else:
            best_reward_so_far = float("inf")

        for g in range(self.generation_num):
            start_time = time.time()

            self.train_or_test = 'train'
            env_train = self.env
            train_Set_setting = env_train.train_Set_setting
            test_Set_setting = env_train.test_Set_setting
            arguments = [(self.device, indi, train_Set_setting, test_Set_setting,
                          self.eval_ep_num, self.ob_rms_mean, self.ob_rms_std,
                          self.processor_num, g, self.config, self.train_or_test,
                          self.dynamic_gamma, self.train_dynamic_gamma,
                          self.dynamic_gamma_wf, self.train_dynamic_gamma_wf) for indi in population]

            start_time_rollout = time.time()
            ctx_type = 'spawn'
            ctx = mp.get_context(ctx_type)
            if len(arguments) == 0:
                print("Error: arguments is empty, check if the population is empty.")
            else:
                if self.processor_num <= 0:
                    raise ValueError("self.processor_num must >= 1")
                else:
                    if self.processor_num > 1:
                        with ctx.Pool(self.processor_num) as p:
                            results = p.map(worker_func, arguments)
                    else:
                        results = [worker_func(arg) for arg in arguments]

            # end rollout
            end_time_rollout = time.time() - start_time_rollout

            # start eval
            start_time_eval = time.time()
            results_df = pd.DataFrame(results).sort_values(by=['policy_id'])

            # Collect PPO trajectory in main process (single-process, with gradients)
            # when the optimizer supports it.  Falls back gracefully for plain ES.
            trajectories = []
            if hasattr(self.optim, 'compute_ppo_surrogate_gradient'):
                trajectories = self._collect_ppo_trajectory(g)

            population, sigma_curr, best_reward_per_g = self.optim.next_population(
                self, results_df, g,
                trajectories=trajectories,
                device=self.device
            )
            end_time_eval = time.time() - start_time_eval

            end_time_generation = time.time() - start_time

            maximization = self.config.config['yaml-config']['optim']['maximization']
            if (maximization and best_reward_per_g > best_reward_so_far) or (not maximization and best_reward_per_g < best_reward_so_far):
                best_reward_so_far = best_reward_per_g

            gamma_per_instance = results_df['gamma_per_instance'].tolist()[0]
            print(f"\ninstances in training: {len(gamma_per_instance)}, gamma_per_instance: {gamma_per_instance}", flush=True)
            print(
                f"episode: {g}, [current_policy_population:], best reward so far: {best_reward_so_far:.4f}, best reward of the current generation: {best_reward_per_g:.4f}, sigma: {sigma_curr:.3f}, time_generation: {end_time_generation:.2f}, rollout_time: {end_time_rollout:.2f}, eval_time: {end_time_eval:.2f}", flush=True
            )
            training_reward, training_VM_cost, training_SLA_penalty = (results_df[col].tolist()[0] for col in ['rewards', 'VM_cost', 'SLA_penalty'])
            print(
                f"episode: {g}, [the_basic_policy:], current training reward: {training_reward:.4f}, current training VM_cost: {training_VM_cost:.4f}, current training SLA_penalty: {training_SLA_penalty:.4f}", flush=True
            )

            # update mean and std every generation
            if self.running_mstd:
                hist_obs = []
                hist_obs = np.concatenate(results_df['hist_obs'], axis=0)
                self.ob_rms.update(hist_obs)
                self.ob_rms_mean = self.ob_rms.mean
                self.ob_rms_std = np.sqrt(self.ob_rms.var)

            if self.log:
                results_df = results_df.drop(['gamma_per_instance'], axis=1)
                if self.running_mstd:
                    results_df = results_df.drop(['hist_obs'], axis=1)
                results_df = results_df.loc[results_df['policy_id'] == -1]

                dir_train = self.save_mode_dir + "/train_performance"
                write_header = not os.path.exists(dir_train)
                if not os.path.exists(dir_train):
                    os.makedirs(dir_train)
                results_df.to_csv(dir_train + "/training_record.csv", index=False, header=write_header, mode='a')

                elite = self.optim.get_elite_model()
                if (g + 1) % self.save_model_freq == 0 or g == 0:
                    if g == 0:
                        save_pth = self.save_mode_dir + "/saved_models" + f"/ep_{g}.pt"
                    else:
                        save_pth = self.save_mode_dir + "/saved_models" + f"/ep_{(g + 1)}.pt"
                    torch.save(elite.state_dict(), save_pth)
                    if self.running_mstd:
                        if g == 0:
                            save_pth = self.save_mode_dir + "/saved_models" + f"/ob_rms_{g}.pickle"
                        else:
                            save_pth = self.save_mode_dir + "/saved_models" + f"/ob_rms_{(g + 1)}.pickle"
                        f = open(save_pth, 'wb')
                        pickle.dump(np.concatenate((self.ob_rms_mean, self.ob_rms_std)), f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.close()

            # evaluate performance on the test set at specified training iteration
            if ((g+1) % self.save_model_freq) == 0 or g == 0:
                from utils.policy_dict import agent_policy
                indi_test = []
                agent_ids_test = self.env.get_agent_ids()
                model_test = self.optim.get_elite_model()
                indi_test.append(agent_policy(agent_ids_test, model_test))
                self.train_or_test = 'test'
                env_test = self.env_test
                train_Set_setting = env_test.train_Set_setting
                test_Set_setting = env_test.test_Set_setting
                arguments = [(self.device, indi, train_Set_setting, test_Set_setting,
                              self.valid_ep_num, self.ob_rms_mean, self.ob_rms_std,
                              self.processor_num, 0, self.config, self.train_or_test,
                              self.dynamic_gamma, self.test_dynamic_gamma,
                              self.dynamic_gamma_wf, self.test_dynamic_gamma_wf) for indi in indi_test]

                # start rollout works
                start_time_test = time.time()
                results = [worker_func(arg) for arg in arguments]
                end_time_test = time.time() - start_time_test
                results_df = pd.DataFrame(results)

                # print testing results in testing process
                testing_reward = results_df['rewards'].tolist()[0]
                VM_cost = results_df["VM_cost"].tolist()[0]
                SLA_penalty = results_df["SLA_penalty"].tolist()[0]
                gamma_per_instance = results_df['gamma_per_instance'].tolist()[0]
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print(f"instances in testing: {len(gamma_per_instance)}, gamma_per_instance: {gamma_per_instance}", flush=True)
                print(
                    f"episode: {g}, [<<<<----testing---->>>>], current testing reward: {testing_reward:.4f}, current testing VM_cost: {VM_cost:.4f}, current testing SLA_penalty: {SLA_penalty:.4f}, current testing_time: {end_time_test:.2f}",
                    flush=True
                )

                if self.log:
                    results_df = results_df.drop(['gamma_per_instance'], axis=1)
                    results_df = results_df.drop(['hist_obs'], axis=1)
                    dir_test = self.save_mode_dir + "/test_performance"
                    write_header = not os.path.exists(dir_test)
                    if not os.path.exists(dir_test):
                        os.makedirs(dir_test)
                    results_df.to_csv(dir_test + "/testing_record_in_training.csv", index=False, header=write_header, mode='a')

    def _collect_ppo_trajectory(self, g):
        """
        Run a single rollout of the elite (parent) policy using *stochastic* action sampling
        to collect a trajectory for the PPO surrogate gradient.

        Key design notes:
        - Runs in the main process (no multiprocessing) so gradients are available.
        - Uses the same training instance index as the current generation (g).
        - ob_rms normalisation is applied identically to worker_func.
        - model.train() enables gradient tracking during the rollout.
        - The value returned from get_value() is detached to a Python float for storage;
          full re-computation happens inside compute_ppo_surrogate_gradient().
        """
        try:
            model = self.optim.get_elite_model()
            if not (hasattr(model, 'model') and hasattr(model.model, 'value_head')):
                return []

            env = builder.build_env(
                self.device, self.config.config,
                self.env.train_Set_setting, self.env.test_Set_setting
            )
            agent_ids = env.get_agent_ids()
            states = env.reset(g, 0, 'train')

            trajectories = []
            done = False
            model.train()

            while not done:
                for agent_id in agent_ids:
                    s = states[agent_id]["state"]
                    dag = states[agent_id]["DAG"]
                    node_id = states[agent_id]["Node_id"]
                    VM_configuration = states[agent_id]["VM_configuration"]
                    sla_gamma = states[agent_id]["sla_gamma"]

                    if s.ndim < 2:
                        s = s[np.newaxis, :]
                    if self.ob_rms_mean is not None:
                        s = (s - self.ob_rms_mean) / self.ob_rms_std

                    # Full forward with Critic; no torch.no_grad() so gradients flow
                    logits, _ = model.model(self.device, s, dag, node_id, VM_configuration)
                    logits = logits.squeeze()
                    if logits.dim() != 1:
                        logits = logits.view(-1)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)

                    dist = torch.distributions.Categorical(logits=logits.float())
                    action_tensor = dist.sample()
                    log_prob = dist.log_prob(action_tensor).item()

                    # Critic value (detached scalar; re-computed inside PPO loss)
                    with torch.no_grad():
                        value_tensor = model.get_value(self.device, s, dag, node_id, VM_configuration)
                        value_scalar = value_tensor.squeeze().item()

                    actions = {agent_id: action_tensor.detach().cpu().numpy()}
                    states, r, done, _ = env.step(actions)

                    trajectories.append({
                        'ob': s,
                        'dag': dag,
                        'node_id': node_id,
                        'VM_features_matrix': VM_configuration,
                        'action': action_tensor.item(),
                        'log_prob': log_prob,
                        'reward': r,
                        'value': value_scalar,
                    })

            model.eval()
            return trajectories

        except Exception as e:
            print(f"[GuidedES-PPO] PPO trajectory collection failed: {e}", flush=True)
            return []

    def eval(self):
        # load policy from log
        self.policy.load_state_dict(torch.load(self.config.config['runtime-config']['policy_path']))
        self.policy.to(self.device)
        self.policy.eval()

        env_test = self.env_test

        indi = agent_policy(env_test.get_agent_ids(), self.policy)

        if self.running_mstd:
            with open(self.config.config['runtime-config']['rms_path'], "rb") as f:
                ob_rms = pickle.load(f)
                self.ob_rms_mean = ob_rms[:int(0.5 * len(ob_rms))]
                self.ob_rms_std = ob_rms[int(0.5 * len(ob_rms)):]

        g = 0
        self.train_or_test = 'test'
        train_Set_setting = env_test.train_Set_setting
        test_Set_setting = env_test.test_Set_setting
        arguments = [(self.device, indi, train_Set_setting, test_Set_setting,
                      self.valid_ep_num, self.ob_rms_mean, self.ob_rms_std,
                      self.processor_num, g, self.config, self.train_or_test,
                      self.dynamic_gamma, self.test_dynamic_gamma,
                      self.dynamic_gamma_wf, self.test_dynamic_gamma_wf)]

        start_time_test = time.time()

        results = [worker_func(arg) for arg in arguments]

        end_time_test = time.time() - start_time_test

        results_df = pd.DataFrame(results)

        testing_reward = results_df['rewards'].tolist()[0]
        VM_cost = results_df["VM_cost"].tolist()[0]
        SLA_penalty = results_df["SLA_penalty"].tolist()[0]
        gamma_per_instance = results_df['gamma_per_instance'].tolist()[0]
        print(f"instances in testing: {len(gamma_per_instance)}, gamma_per_instance: {gamma_per_instance}", flush=True)
        print(
            f"current testing reward: {testing_reward:.4f}, current VM cost: {VM_cost:.4f}, current SLA penalty: {SLA_penalty:.4f}, testing_time: {end_time_test:.2f}\n", flush=True
        )

        if self.log:
            results_df = results_df.drop(['gamma_per_instance'], axis=1)
            results_df = results_df.drop(['hist_obs'], axis=1)
            if self.dynamic_gamma:
                if self.config.config['runtime-config']['final'] == True:
                    dir_test = os.path.dirname(self.config.config['runtime-config']['config']) + "/test_performance_dynamicGamma_final"
                    test_size = self.config.config['yaml-config']['env']['wf_size']
                    model_num = self.config.config['yaml-config']['env']['model_num']
                    csv_path = dir_test + "/testing_record" + "_" + str(test_size) + "_" + str(model_num) + ".csv"
                    write_header = not os.path.exists(csv_path)
                    if not os.path.exists(dir_test):
                        os.makedirs(dir_test)
                    results_df.to_csv(csv_path, index=False, header=write_header, mode='a')
                else:
                    dir_test = os.path.dirname(self.config.config['runtime-config']['config']) + "/test_performance_dynamicGamma"
                    test_size = self.config.config['yaml-config']['env']['wf_size']
                    csv_path = dir_test + "/testing_record" + "_" + str(test_size) + ".csv"
                    write_header = not os.path.exists(csv_path)
                    if not os.path.exists(dir_test):
                        os.makedirs(dir_test)
                    results_df.to_csv(csv_path, index=False, header=write_header, mode='a')
            else:
                if self.config.config['runtime-config']['final'] == True:
                    dir_test = os.path.dirname(self.config.config['runtime-config']['config']) + "/test_performance_final"
                    test_size = self.config.config['yaml-config']['env']['wf_size']
                    gamma_size = self.config.config['yaml-config']['env']['gamma_test']
                    model_num = self.config.config['yaml-config']['env']['model_num']
                    csv_path = dir_test + "/testing_record_"+str(gamma_size)+"_"+str(test_size)+"_"+str(model_num)+".csv"
                    write_header = not os.path.exists(csv_path)
                    if not os.path.exists(dir_test):
                        os.makedirs(dir_test)
                    results_df.to_csv(csv_path, index=False, header=write_header, mode='a')
                else:
                    dir_test = os.path.dirname(self.config.config['runtime-config']['config']) + "/test_performance"
                    test_size = self.config.config['yaml-config']['env']['wf_size']
                    gamma_size = self.config.config['yaml-config']['env']['gamma_test']
                    csv_path = dir_test + "/testing_record_"+str(gamma_size)+"_"+str(test_size)+".csv"
                    write_header = not os.path.exists(csv_path)
                    if not os.path.exists(dir_test):
                        os.makedirs(dir_test)
                    results_df.to_csv(csv_path, index=False, header=write_header, mode='a')


def worker_func(arguments):
    (device, indi, train_Set_setting, test_Set_setting, eval_ep_num, ob_rms_mean, ob_rms_std, processor_num, g, config,
     train_or_test, dynamic_gamma, dynamic_gamma_matrix, dynamic_gamma_wf, dynamic_gamma_matrix_wf) = arguments
    if train_or_test == 'train':
        config.config['yaml-config']['env']['gamma'] = config.config['yaml-config']['env']['gamma_train']
    elif train_or_test == 'test':
        config.config['yaml-config']['env']['gamma'] = config.config['yaml-config']['env']['gamma_test']
    env = builder.build_env(device, config.config, train_Set_setting, test_Set_setting)

    hist_rewards = {}
    hist_obs = {}
    hist_actions = {}
    obs = None
    total_reward = 0

    total_VM_execHour = 0
    total_VM_totHour = 0
    total_VM_cost = 0
    total_SLA_penalty = 0
    total_missDeadlineNum = 0
    gamma_per_instance = []

    for ep_num in range(eval_ep_num):
        if dynamic_gamma and not dynamic_gamma_wf:
            env.set.gamma = dynamic_gamma_matrix[g][ep_num]
        elif dynamic_gamma and dynamic_gamma_wf:
            env.set.gamma = dynamic_gamma_matrix_wf[g][ep_num]
        gamma_per_instance.append(env.set.gamma)

        states = env.reset(g, ep_num, train_or_test)

        rewards_per_eval = []
        obs_per_eval = []
        actions_per_eval = []
        done = False

        for agent_id, model in indi.items():
            model.eval()
            model.reset()
        while not done:
            actions = {}
            for agent_id, model in indi.items():
                s = states[agent_id]["state"]
                dag = states[agent_id]["DAG"]
                node_id = states[agent_id]["Node_id"]
                VM_configuration = states[agent_id]["VM_configuration"]
                sla_gamma = states[agent_id]["sla_gamma"]
                if s.ndim < 2:
                    s = s[np.newaxis, :]
                if ob_rms_mean is not None:
                    s = (s - ob_rms_mean) / ob_rms_std

                if "removeVM" in states:
                    with torch.no_grad():
                        actions[agent_id], _ = model(device, s, dag, node_id, VM_configuration, sla_gamma, removeVM=states["removeVM"])
                else:
                    with torch.no_grad():
                        actions[agent_id], _ = model(device, s, dag, node_id, VM_configuration,  sla_gamma)

                states, r, done, _ = env.step(actions)

                rewards_per_eval.append(r)
                obs_per_eval.append(s)
                actions_per_eval.append(actions[agent_id])
                total_reward += r

                if obs is None:
                    obs = states["0"]["state"]
                else:
                    obs = np.append(obs, states["0"]["state"], axis=0)

        hist_rewards[ep_num] = rewards_per_eval
        hist_obs[ep_num] = obs_per_eval
        hist_actions[ep_num] = actions_per_eval

        total_VM_execHour += env.episode_info["VM_execHour"]
        total_VM_totHour += env.episode_info["VM_totHour"]
        total_VM_cost += env.episode_info["VM_cost"]
        total_SLA_penalty += env.episode_info["SLA_penalty"]
        total_missDeadlineNum += env.episode_info["missDeadlineNum"]

    rewards_mean = total_reward / eval_ep_num

    VM_execHour_mean = total_VM_execHour / eval_ep_num
    VM_totHour_mean = total_VM_totHour / eval_ep_num
    VM_cost_mean = total_VM_cost / eval_ep_num
    SLA_penalty_mean = total_SLA_penalty / eval_ep_num
    missDeadlineNum_mean = total_missDeadlineNum / eval_ep_num

    if True:
        if indi['0'].policy_id == -1:
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "VM_execHour": VM_execHour_mean,
                    "VM_totHour": VM_totHour_mean,
                    "VM_cost": VM_cost_mean,
                    "SLA_penalty": SLA_penalty_mean,
                    "missDeadlineNum": missDeadlineNum_mean,
                    "gamma_per_instance": gamma_per_instance}
        else:  # we do not record detailed info for non-parent policy
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    'hist_obs': obs,
                    "VM_execHour": np.nan,
                    "VM_totHour": np.nan,
                    "VM_cost": np.nan,
                    "SLA_penalty": np.nan,
                    "missDeadlineNum": np.nan,
                    "gamma_per_instance": np.nan}


def discount_rewards(rewards):
    gamma = 0.99
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards
