
"""
Algorithm: SPN_CWS
Paper: Cost-Aware Dynamic Cloud Workflow Scheduling Using Self-attention and Evolutionary Reinforcement Learning. ICSOC 2024.
Authors: Ya Shen, Gang Chen, Hui Ma, and Mengjie Zhang
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policy.base_model import BasePolicy
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")

class SelfAttentionEncoder(nn.Module):
    def __init__(self, task_fea_size, vm_fea_size, output_size,
                 d_model, num_heads, num_en_layers, d_ff, dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()
        # Task task_preprocess
        self.task_embedding = nn.Sequential(nn.Linear(task_fea_size, d_model))
        self.task_feature_enhance = nn.Sequential(nn.Linear(d_model, 2*d_model),
                                                  nn.ReLU(),
                                                  nn.Linear(2*d_model, d_model))

        # VM task_preprocess
        self.vm_embedding = nn.Sequential(nn.Linear(vm_fea_size, d_model))

        # self-attention
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_en_layers)

        # priority mapping
        self.priority = nn.Sequential(nn.Linear(2 * d_model, 4 * d_model),
                                      nn.ReLU(),
                                      nn.Linear(4 * d_model, 4 * d_model),
                                      nn.ReLU(),
                                      nn.Linear(4 * d_model, output_size))

    def forward(self, device, ob, dag, node_id, VM_features_matrix):
        ob = torch.from_numpy(ob.astype(np.float32)).to(device)
        VM_features_matrix = torch.from_numpy(VM_features_matrix.astype(np.float32)).to(device)

        task_info = ob[:, 0:-4].unsqueeze(1)
        vm_info = ob[:, -4::].unsqueeze(1)

        # Task task_preprocess
        task_embedded = self.task_embedding(task_info)
        task_feature_enhance = self.task_feature_enhance(task_embedded)

        # VM task_preprocess
        vm_embedded = self.vm_embedding(vm_info)

        # self-attention
        vm_embedded = vm_embedded.permute(1, 0, 2)
        global_info = self.transformer_encoder(vm_embedded)
        global_info = global_info.permute(1, 0, 2)

        # Feature concatenation
        concatenation_features = torch.cat((global_info, task_feature_enhance), dim=-1)

        # priority mapping
        priority = self.priority(concatenation_features)

        return priority


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

        self.model = SelfAttentionEncoder(task_fea_size=4, vm_fea_size=4, output_size=1, d_model=16,
                                          num_heads=2, num_en_layers=2, d_ff=64, dropout=0.1)

    def forward(self, device, ob, dag, node_id, VM_features_matrix, sla_gamma, removeVM=None):
        out = self.model(device, ob, dag, node_id, VM_features_matrix)

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
