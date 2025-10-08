'''
No contrastive MI_max (use classification for MI max) ['Cora', 'sector_cat', 3, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/text.GraphMAE.GCN.128.2.pth']
['task_prompt', ['label', 'label', 'label', 'label', 'label']]
['ACC', 0.6777, 0.0191]
['Macro_F1', 0.661, 0.0222]
['Micro_F1', 0.6777, 0.0191]
['Weighted_F1', 0.6753, 0.0169]
['ROC', 0.9094, 0.0178]
['PRC', 0.6573, 0.0339]

['Cora', 'sector_cat', 3, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/text.GraphMAE.GCN.128.2.pth']
['task_prompt', ['label', 'label', 'label', 'label', 'label']]
['ACC', 0.7058, 0.0362]
['Macro_F1', 0.6918, 0.0389]
['Micro_F1', 0.7058, 0.0362]
['Weighted_F1', 0.7073, 0.0345]
['ROC', 0.9237, 0.0132]
['PRC', 0.7496, 0.045]

['Cora', 'sector_cat', 1, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/text.GraphMAE.GCN.128.2.pth']
['task_prompt', ['label', 'label', 'label', 'label', 'label']]
['ACC', 0.6003, 0.0852]
['Macro_F1', 0.591, 0.0771]
['Micro_F1', 0.6003, 0.0852]
['Weighted_F1', 0.6014, 0.0889]
['ROC', 0.886, 0.0325]
['PRC', 0.6623, 0.064]

['Cora', 'sector_cat', 3, 3, '/home/jz875/project/label_prompt/pre_trained_model/Cora/text.GraphMAE.GCN.128.2.pth']
['task_prompt', ['label', 'label', 'label', 'label', 'label']]
['ACC', 0.6974, 0.0255]
['Macro_F1', 0.6925, 0.0232]
['Micro_F1', 0.6974, 0.0255]
['Weighted_F1', 0.6925, 0.0232]
['ROC', 0.8572, 0.0144]
['PRC', 0.7536, 0.0246]

['Cora', 'sector_cat', 1, 3, '/home/jz875/project/label_prompt/pre_trained_model/Cora/text.GraphMAE.GCN.128.2.pth']
['task_prompt', ['label', 'label', 'label', 'label', 'label']]
['ACC', 0.5881, 0.0921]
['Macro_F1', 0.5746, 0.0962]
['Micro_F1', 0.5881, 0.0921]
['Weighted_F1', 0.5746, 0.0962]
['ROC', 0.762, 0.0799]
['PRC', 0.6132, 0.1162]

['Computers', 'sector_cat', 3, 10, '/home/jz875/project/label_prompt/pre_trained_model/Computers/text.GraphMAE.GCN.128.2.pth']
['task_prompt', ['label', 'label', 'label', 'label', 'label']]
['ACC', 0.6346, 0.0751]
['Macro_F1', 0.6201, 0.0679]
['Micro_F1', 0.6346, 0.0751]
['Weighted_F1', 0.5949, 0.0887]
['ROC', 0.9329, 0.0218]
['PRC', 0.754, 0.03]

['Computers', 'sector_cat', 1, 10, '/home/jz875/project/label_prompt/pre_trained_model/Computers/text.GraphMAE.GCN.128.2.pth']
['task_prompt', ['label', 'label', 'label', 'label', 'label']]
['ACC', 0.5362, 0.0711]
['Macro_F1', 0.5029, 0.0873]
['Micro_F1', 0.5362, 0.0711]
['Weighted_F1', 0.5111, 0.0851]
['ROC', 0.8693, 0.0335]
['PRC', 0.6011, 0.0578]
'''

import torch
import random
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_random_edge, dropout_edge, dropout_adj
from torch_geometric.nn import SimpleConv, GCNConv, GATConv, AGNNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchmetrics


class label_prompt_node(torch.nn.Module):
    def __init__(self, feature_dim, in_channels, n_classes, dataset, device):
        super(label_prompt_node, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.dataset = dataset
        self.device = device
        
        self.feature_prompt = torch.nn.Parameter(torch.Tensor(1, feature_dim))
        self.propagate_prompt = torch.nn.Parameter(torch.Tensor(1, 2*feature_dim))
        self.readout_prompt = torch.nn.Parameter(torch.Tensor(1, in_channels))

        if self.dataset in ['PubMed']:
            self.aggregate_function = global_max_pool
        if self.dataset in ['squirrel']:
            self.aggregate_function = global_add_pool
        if self.dataset in ['Photo']:
            self.aggregate_function = global_mean_pool
        if self.dataset in ['Computers']:
            self.aggregate_function = global_mean_pool

        if self.dataset in ['PubMed']:
            self.perturb_times = 5
        if self.dataset in ['squirrel']:
            self.perturb_times = 5
        if self.dataset in ['Photo']:
            self.perturb_times = 10
        if self.dataset in ['Computers']:
            self.perturb_times = 5

        self.answering = nn.Linear(in_channels, n_classes)
        self.noise_matrix = nn.Linear(feature_dim, feature_dim)
        
        self.loss = nn.CrossEntropyLoss()
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.feature_prompt)
        torch.nn.init.xavier_uniform_(self.propagate_prompt)
        torch.nn.init.xavier_uniform_(self.readout_prompt)
        torch.nn.init.xavier_uniform_(self.noise_matrix.weight)
        torch.nn.init.xavier_uniform_(self.answering.weight)

    def get_critical_emb(self, gnn, x, edge_index, batch = None):
        if self.dataset in ['PubMed']:
            new_x = 0.1*self.feature_prompt + x
            tmp_emb = gnn(new_x, edge_index)*self.readout_prompt
        
        if self.dataset in ['squirrel']:
            new_x = 0.1*self.feature_prompt + x
            tmp_emb = gnn(new_x, edge_index)*self.readout_prompt
        
        if self.dataset in ['Photo']:
            new_x = self.feature_prompt + x
            tmp_emb = gnn(new_x, edge_index)*self.readout_prompt
        
        if self.dataset in ['Computers']:
            new_x = 0.1*self.feature_prompt + x
            tmp_emb = gnn(new_x, edge_index)*self.readout_prompt

        return self.aggregate_function(tmp_emb, batch)
        
    def get_boundary_samples(self, data):
        if self.dataset in ['PubMed']:
            tmp_x = data.x + 0.1*F.normalize(torch.tanh(self.noise_matrix(data.x)), 1, -1)
            tmp_edge_index, _ = dropout_edge(data.edge_index, p = 0.1)
        
        if self.dataset in ['squirrel']:
            tmp_x = data.x + 0.1*F.normalize(torch.tanh(self.noise_matrix(data.x)), 1, -1)
            tmp_edge_index, _ = dropout_edge(data.edge_index, p = 0.1)
        
        if self.dataset in ['Photo']:
            tmp_x = data.x + F.normalize(F.relu(self.noise_matrix(data.x)), 1, -1)
            tmp_edge_index, _ = dropout_edge(data.edge_index, p = 0.1)
        
        if self.dataset in ['Computers']:
            tmp_x = data.x + F.normalize(torch.tanh(self.noise_matrix(data.x)), 1, -1)
            tmp_edge_index, _ = dropout_edge(data.edge_index, p = 0.1)

        return tmp_x, tmp_edge_index, data.batch

    def MI_loss(self, gnn, data):
        # max_MI (z|y)
        critical_embedding_sample = self.get_critical_emb(gnn, data.x, data.edge_index, data.batch) # batch_size | hidden_dim
        max_loss = self.loss(self.answering(critical_embedding_sample), data.y)
        
        # generate boundary samples and get necessary features
        boundary_features_list = []
        boundary_y_list = []
        critical_embedding_boundary_list = []
        for i in range(self.perturb_times):
            # generate boundary samples
            boundary_x, boundary_edge_index, boundary_batch = self.get_boundary_samples(data)
            critical_embedding_boundary = self.get_critical_emb(gnn, boundary_x, boundary_edge_index, data.batch)
            max_loss += self.loss(self.answering(critical_embedding_boundary), data.y) # max_MI (z|y)
            critical_embedding_boundary_list.append(critical_embedding_boundary)

            tmp_boundary_feature = gnn(boundary_x, boundary_edge_index)
            tmp_boundary_feature = self.aggregate_function(tmp_boundary_feature, boundary_batch)
            boundary_features_list.append(tmp_boundary_feature)
            boundary_y_list.append(data.y)
        
        boundary_features = torch.stack(boundary_features_list, dim = 0)
        boundary_features_critical = torch.stack(critical_embedding_boundary_list, dim = 0) # perturb_times | shot_num | emb_dim

        # keep boundary samples
        boundary_features = boundary_features.view(data.y.size(0), self.perturb_times, -1, self.in_channels) # batch_size | perturb_times | shot_num | emb_dim
        boundary_features_p = boundary_features[torch.randperm(boundary_features.size(0)), :, :, :]
        boundary_features_p = boundary_features_p[:, torch.randperm(boundary_features_p.size(1)), :, :]
        boundary_features_n = boundary_features[:, torch.randperm(boundary_features.size(1)), :, :]
        
        pos_distance = torch.norm(boundary_features_p - boundary_features, 2, -1)
        neg_distance = torch.norm(boundary_features_n - boundary_features, 2, -1)
       
        if self.dataset in ['PubMed']:
            noise_loss = 0.1*torch.sigmoid(-neg_distance).mean() + F.relu(neg_distance-pos_distance).mean() # 使同一类别内的干扰样本尽可能分散，从而触及边界; 同时同一类别的扰动样本距离应小于不同类别扰动样本
        
        if self.dataset in ['squirrel']:
            noise_loss = 0.1*torch.sigmoid(-neg_distance).mean() + F.relu(neg_distance-pos_distance).mean() # 使同一类别内的干扰样本尽可能分散，从而触及边界; 同时同一类别的扰动样本距离应小于不同类别扰动样本
        
        if self.dataset in ['Photo']:
            noise_loss = 0.1*torch.sigmoid(-neg_distance).mean() + F.relu(neg_distance-pos_distance).mean() # 使同一类别内的干扰样本尽可能分散，从而触及边界; 同时同一类别的扰动样本距离应小于不同类别扰动样本
        
        if self.dataset in ['Computers']:
            noise_loss = torch.sigmoid(-neg_distance).mean() + F.relu(neg_distance-pos_distance).mean() # 使同一类别内的干扰样本尽可能分散，从而触及边界; 同时同一类别的扰动样本距离应小于不同类别扰动样本
        
        # min_MI
        boundary_features_critical = boundary_features_critical.view(data.y.size(0), self.perturb_times, -1, self.in_channels) # batch_size(n_classes) | perturb_times | shot_num | emb_dim
        boundary_features_critical_p = boundary_features_critical[:, torch.randperm(boundary_features_critical.size(1)), :, :]

        pos_distance = torch.norm(boundary_features_critical_p - boundary_features_critical, 2, -1)
        neg_distance = torch.norm(boundary_features - boundary_features_critical, 2, -1)
        
        if self.dataset in ['PubMed']:
            min_loss = F.relu(neg_distance-pos_distance).mean()
        
        if self.dataset in ['squirrel']:
            min_loss = F.relu(neg_distance-pos_distance).mean()
        
        if self.dataset in ['Photo']:
            min_loss = F.relu(neg_distance-pos_distance+1).mean()
        
        if self.dataset in ['Computers']:
            min_loss = F.relu(neg_distance-pos_distance).mean()

        
        return max_loss, min_loss, noise_loss
        
    
    def Eva_Node(self, loader, data, idx_test, idx_train, gnn, answering, num_class, device, center = None): 
        gnn.eval()
        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
        micro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="micro").to(device)
        weighted_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="weighted").to(device)
        auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
        auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

        accuracy.reset()
        macro_f1.reset()
        micro_f1.reset()
        weighted_f1.reset()
        auroc.reset()
        auprc.reset()

        with torch.no_grad(): 
            for batch_id, batch in enumerate(loader): 
                batch = batch.to(device) 
                critical_emb = self.get_critical_emb(gnn, batch.x, batch.edge_index, batch.batch)
                out = self.answering(critical_emb)
                pred = out.argmax(dim=1)

                acc = accuracy(pred, batch.y)
                ma_f1 = macro_f1(pred, batch.y)
                mi_f1 = micro_f1(pred, batch.y)
                wei_f1 = weighted_f1(pred, batch.y)
                roc = auroc(out, batch.y)
                prc = auprc(out, batch.y)
        acc = accuracy.compute()
        macro = macro_f1.compute()
        micro = micro_f1.compute()
        weighted = weighted_f1.compute()
        roc = auroc.compute()
        prc = auprc.compute()
    
        return {'task_prompt': 'label', 'ACC':acc.item(), 'Macro_F1':macro.item(), 'Micro_F1':micro.item(), 'Weighted_F1':weighted.item(), 'ROC':roc.item(), 'PRC':prc.item()}