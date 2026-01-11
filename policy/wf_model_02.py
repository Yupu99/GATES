
"""
Algorithm: GATES
Paper: GATES: Cost-aware Dynamic Workflow Scheduling via Graph Attention Networks and Evolution Strategy. IJCAI 2025.
Authors: Ya Shen, Gang Chen, Hui Ma, and Mengjie Zhang
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policy.base_model import BasePolicy
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")

class SelfAttentionEncoder(nn.Module):
    def __init__(self,
                 task_fea_size,
                 vm_fea_size,
                 output_size,
                 d_model,
                 att_heads,
                 att_en_layers,
                 d_ff,
                 gat_heads,
                 dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()

        # Task preprocess
        self.task_feature_enhance = nn.Sequential(nn.Linear(task_fea_size, 128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, d_model))

        # VM preprocess
        self.vm_embedding = nn.Sequential(nn.Linear(vm_fea_size, d_model))

        # self-attention
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, att_heads, d_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, att_en_layers)

        # GATs
        self.gat_dag_layer01 = GATConv(6, d_model, heads=gat_heads, concat=True)
        self.gat_dag_layer02 = GATConv(gat_heads*d_model, d_model, heads=1, concat=False)
        self.gat_vm_layer01 = GATConv(4, d_model, heads=gat_heads, concat=True)
        self.gat_vm_layer02 = GATConv(gat_heads*d_model, d_model, heads=1, concat=False)

        # priority mapping
        self.priority = nn.Sequential(nn.Linear(6 * d_model, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, output_size))

    def forward(self, device, ob, dag, node_id, VM_features_matrix):
        """
            ----------1)workflow_embedded + vm_global_info
        """
        ob = torch.from_numpy(ob.astype(np.float32)).to(device)
        VM_features_matrix = torch.from_numpy(VM_features_matrix.astype(np.float32)).to(device)

        readyTask_info = ob[0, 0:-4].unsqueeze(0)
        vm_info = ob[:, -4::].unsqueeze(1)

        workflow_embedded = self.task_feature_enhance(readyTask_info)

        vm_embedded = self.vm_embedding(vm_info)
        vm_embedded = vm_embedded.permute(1, 0, 2)
        vm_global_info = self.transformer_encoder(vm_embedded)
        vm_global_info = vm_global_info.permute(1, 0, 2).squeeze(1)

        """
            ----------2)GAT for tasks DAG
        """
        adj_matrix = nx.adjacency_matrix(dag).todense()
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)

        node_features = []
        predecessor = adj_matrix.sum(dim=0)
        successor = adj_matrix.sum(dim=1)
        for node, data in dag.nodes(data=True):
            # predecessor, successor, processTime, size, sub_deadline, scheduled
            if node == node_id:  # the "scheduled" of readyTask is 2.0
                fea = [predecessor[node], successor[node], data["processTime"], data["size"], data["sub_deadline"], 2.0]
                node_features.append(fea)
            else:
                fea = [predecessor[node], successor[node], data["processTime"], data["size"], data["sub_deadline"], data["scheduled"]]
                node_features.append(fea)
        node_features = torch.tensor(node_features, dtype=torch.float32).to(device)

        mean = node_features.mean(dim=0, keepdim=True)
        std = node_features.std(dim=0, keepdim=True)
        node_features = (node_features - mean) / std

        # Prepare the data for GAT_dag
        edge_index = adj_matrix.nonzero(as_tuple=False).t()
        edge_index_reversed = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, edge_index_reversed], dim=1).to(device)
        data_for_GAT_dag = Data(x=node_features, edge_index=edge_index)

        # GAT_dag process the data
        dag_x, dag_edge_index = data_for_GAT_dag.x, data_for_GAT_dag.edge_index
        dag_x = F.elu(self.gat_dag_layer01(dag_x, dag_edge_index))
        dag_x = F.elu(self.gat_dag_layer02(dag_x, dag_edge_index))
        dag_x_mean = dag_x.mean(dim=0, keepdim=True)
        dag_x_ready = dag_x[node_id].unsqueeze(0)

        """
            ----------3)GAT for tasks-VM graph
        """
        readyTask_vm_features = torch.cat((readyTask_info, vm_info.squeeze(1)), dim=0).to(device)

        num_nodes = readyTask_vm_features.shape[0]
        target_nodes = torch.zeros(num_nodes, dtype=torch.long)
        neighbor_nodes = torch.arange(0, num_nodes, dtype=torch.long)
        edge_forward = torch.stack([target_nodes, neighbor_nodes], dim=0)
        edge_backward = torch.stack([neighbor_nodes, target_nodes], dim=0)
        taskVM_edge_index = torch.cat([edge_forward, edge_backward], dim=1).to(device)

        data_for_GAT_vm = Data(x=readyTask_vm_features, edge_index=taskVM_edge_index)

        taskVM_x, taskVM_edge_index = data_for_GAT_vm.x, data_for_GAT_vm.edge_index
        taskVM_x = F.elu(self.gat_vm_layer01(taskVM_x, taskVM_edge_index))
        taskVM_x = F.elu(self.gat_vm_layer02(taskVM_x, taskVM_edge_index))
        taskVM_x_ready = taskVM_x[0].unsqueeze(0)

        """
            ----------4)----------
        """
        state_embedding = torch.cat((workflow_embedded, dag_x_mean, dag_x_ready, taskVM_x_ready), dim=-1)
        state_embedding = state_embedding.expand(vm_global_info.shape[0], -1)
        state_embedding = torch.cat((state_embedding, taskVM_x[1:]), dim=-1)
        concatenation_features = torch.cat((vm_global_info, state_embedding), dim=-1)
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

        # Policy networks
        self.model = SelfAttentionEncoder(task_fea_size=4, vm_fea_size=4, output_size=1, d_model=16,
                                          att_heads=2, att_en_layers=2, d_ff=128, gat_heads=2, dropout=0.1)

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
