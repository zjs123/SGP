import torch
import numpy as np
from task_strategies.label_prompt_v1 import label_prompt_v1
from task_strategies.MLP_finetune import MLP_finetune
from task_strategies.GPF import GPF_plus, GPF
from task_strategies.Graphprompt import Gprompt, Gprompt_tuning_loss
from task_strategies.GPPT import GPPTPrompt
from task_strategies.All_in_one import HeavyPrompt
from torch import nn, optim

import sys
sys.path.append("..")
from models import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer

class BaseTask:
    def __init__(self, pre_train_model_path='None', gnn_type='TransformerConv',
                 hid_dim = 128, num_layer = 2, dataset_name='Cora', task_train_type='MLP_finetune', epochs=100, shot_num=10, device : int = 5, lr =0.001, wd = 5e-4,
                 batch_size = 16, downstream_task='None', search = False, feature_type='text'):
        
        self.pre_train_model_path = pre_train_model_path
        self.pre_train_type = self.return_pre_train_type(pre_train_model_path)
        self.device = torch.device('cuda:'+ str(device) if torch.cuda.is_available() else 'cpu') if device in [0,1,2,3] else torch.device('cpu')
        self.hid_dim = hid_dim
        self.num_layer = num_layer
        self.dataset_name = dataset_name
        self.shot_num = shot_num
        self.gnn_type = gnn_type
        self.task_train_type = task_train_type
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.search = search
        self.downstream_task = downstream_task
        self.initialize_lossfn()

    def initialize_lossfn(self):
        print(self.downstream_task)
        if self.task_train_type == 'Gprompt':
            self.criterion = Gprompt_tuning_loss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def initialize_optimizer(self):
        model_param_group = []
        if '.pth' not in self.pre_train_model_path:
            model_param_group.append({"params": self.gnn.parameters()})
            model_param_group.append({"params": self.answering.parameters()})
            self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
        else:
            if self.task_train_type == 'MLP_finetune':
                self.optimizer = optim.Adam(self.answering.parameters(), lr=self.lr, weight_decay=self.wd)
            elif self.task_train_type in ['GPF', 'GPF-plus']:
                model_param_group.append({"params": self.prompt.parameters()})
                model_param_group.append({"params": self.answering.parameters()})
                self.optimizer = optim.Adam(model_param_group, lr=self.lr, weight_decay=self.wd)
            elif self.task_train_type in ['label_prompt']:
                self.optimizer = optim.Adam(self.prompt.parameters(), lr=self.lr, weight_decay=self.wd)
            elif self.task_train_type in ['Gprompt']:
                self.pg_opi = optim.Adam(self.prompt.parameters(), lr=self.lr, weight_decay=self.wd)
            elif self.task_train_type in ['GPPT']:
                self.pg_opi = optim.Adam(self.prompt.parameters(), lr=self.lr, weight_decay=self.wd)
            elif self.task_train_type == 'All-in-one':
                self.pg_opi = optim.Adam( self.prompt.parameters(), lr=1e-6, weight_decay= self.wd)
                self.answer_opi = optim.Adam( self.answering.parameters(), lr=self.lr, weight_decay= self.wd)

    def initialize_prompt(self):
        if self.task_train_type == 'MLP_finetune':
            self.prompt = MLP_finetune(self.input_dim).to(self.device)
        elif self.task_train_type == 'label_prompt':
            self.prompt = label_prompt_v1(self.hid_dim, self.output_dim, self.device).to(self.device)
        elif self.task_train_type == 'GPF':
            self.prompt = GPF(self.input_dim).to(self.device)
        elif self.task_train_type == 'GPF-plus':
            self.prompt = GPF_plus(self.input_dim, 20).to(self.device)
        elif self.task_train_type == 'Gprompt':
            self.prompt = Gprompt(self.hid_dim).to(self.device)
        elif self.task_train_type == 'GPPT':
            self.prompt = GPPTPrompt(self.hid_dim, self.output_dim, self.output_dim, device = self.device)
        elif self.task_train_type =='All-in-one':
            self.prompt = HeavyPrompt(token_dim=self.input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3).to(self.device)

    def initialize_gnn(self):
        if '.pth' in self.pre_train_model_path:
            if self.gnn_type not in self.pre_train_model_path :
                raise ValueError(f"the Downstream gnn '{self.gnn_type}' does not match the pre-train model")
            if self.dataset_name not in self.pre_train_model_path :
                raise ValueError(f"the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset")

            self.gnn = torch.load(self.pre_train_model_path, map_location='cpu')   
            print("Successfully loaded pre-trained weights!")
        else:
            if self.gnn_type == 'GAT':
                self.gnn = GAT(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
            elif self.gnn_type == 'GCN':
                self.gnn = GCN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
            elif self.gnn_type == 'GraphSAGE':
                self.gnn = GraphSAGE(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
            elif self.gnn_type == 'GIN':
                self.gnn = GIN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
            elif self.gnn_type == 'GCov':
                self.gnn = GCov(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
            elif self.gnn_type == 'GraphTransformer':
                self.gnn = GraphTransformer(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        self.gnn.to(self.device)
        print(self.gnn)


    def return_pre_train_type(self, pre_train_model_path):
        names = ['LP', 'DGI', 'GraphMAE','Edgepred_GPPT', 'Edgepred_Gprompt','GraphCL', 'SimGRACE']
        for name in names:
            if name  in  pre_train_model_path:
                return name


      
 
            
      