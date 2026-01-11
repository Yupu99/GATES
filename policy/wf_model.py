
"""
Paper: Cost-aware dynamic multi-workflow scheduling in cloud data center using evolutionary reinforcement learning. ICSOC 2022.
Authors: Victoria Huang, Chen Wang, Hui Ma, Gang Chen, and Kameron Christopher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policy.base_model import BasePolicy
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")


class SelfAttentionEncoder(nn.Module):
    def __init__(self):
        super(SelfAttentionEncoder, self).__init__()
        self.priority = nn.Sequential(nn.Linear(8, 64),
                                   nn.Tanh(),  # tanh in Victoria's paper
                                   nn.Linear(64, 64),
                                   nn.Tanh(),
                                   nn.Linear(64, 1))

    def forward(self, ob):
        x = self.priority(ob)

        return x


class WFPolicy(BasePolicy):
    def __init__(self, config, policy_id=-1):
        super(WFPolicy, self).__init__()
        self.config = config
        self.policy_id = policy_id
        self.state_num = config['state_num']
        self.action_num = config['action_num']
        self.discrete_action = config['discrete_action']
        if "add_gru" in config:
            self.add_gru = config['add_gru']
        else:
            self.add_gru = True

        self.fc1 = nn.Linear(self.state_num, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_num)

        self.model = SelfAttentionEncoder()

    def forward(self, device, ob, dag, node_id, VM_features_matrix, sla_gamma, removeVM=None):
        ob = torch.from_numpy(ob.astype(np.float32)).to(device)
        x = ob.unsqueeze(0)  # Todo: check x dim as its condition
        out = self.model(x)

        logits = out.squeeze().to(device)
        if logits.dim() != 1:
            logits = logits.view(-1)

        if removeVM is not None:
            idx = torch.as_tensor(list(removeVM) if isinstance(removeVM, (list, tuple, set, np.ndarray, torch.Tensor)) else [removeVM], device=device, dtype=torch.long)
            logits[idx] = float("-inf")

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
        if torch.isinf(logits).all() and (logits < 0).all():
            logits = torch.zeros_like(logits)

        if self.discrete_action:
            if self.config['greedy_action']:
                action = torch.argmax(logits)
            else:
                with torch.amp.autocast('cuda', enabled=False):
                    dist = torch.distributions.Categorical(logits=logits.float())
                    action = dist.sample()
        else:
            action = torch.relu(logits)

        action_np = action.detach().cpu().numpy()
        out_1xA = logits.view(1, -1)
        return action_np, out_1xA

    def xavier_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def zero_init(self):
        for param in self.parameters():
            param.data = torch.zeros(param.shape)

    def norm_init(self, std=1.0):
        for param in self.parameters():
            shape = param.shape
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            param.data = torch.from_numpy(out)

    def set_policy_id(self, policy_id):
        self.policy_id = policy_id

    def reset(self):
        pass

    def get_param_list(self):
        param_lst = []
        for param in self.parameters():
            param_lst.append(param.data.numpy())
        return param_lst

    def set_param_list(self, param_lst: list):
        lst_idx = 0
        for param in self.parameters():
            param.data = torch.tensor(param_lst[lst_idx]).float()
            lst_idx += 1
