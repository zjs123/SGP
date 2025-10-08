from pre_train_strategies.GraphMAE import GraphMAE
from pre_train_strategies.GCL import GraphCL
from pre_train_strategies.DGI import DGI
from pre_train_strategies.LP import LP
from get_args import get_pretrain_args, get_task_args

import os
import numpy as np
import random
import torch
from random import shuffle

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(1024)
args = get_pretrain_args()

for data_set in ['COX2', 'BZR', 'MUTAG', 'COLLAB']:
    for task in ["DGI", "GraphCL"]:
        args.dataset_name = data_set
        args.pretrain_task = task
        if args.pretrain_task == 'LP':
            pt = LP(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, pretrain_task=args.pretrain_task, feature_type=args.feature_type)
        if args.pretrain_task == 'SimGRACE':
            pt = SimGRACE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, pretrain_task=args.pretrain_task, feature_type=args.feature_type)
        if args.pretrain_task == 'GraphCL':
            pt = GraphCL(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, pretrain_task=args.pretrain_task, feature_type=args.feature_type)
        if args.pretrain_task == 'Edgepred_GPPT':
            pt = Edgepred_GPPT(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, pretrain_task=args.pretrain_task, feature_type=args.feature_type)
        if args.pretrain_task == 'Edgepred_Gprompt':
            pt = Edgepred_Gprompt(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, pretrain_task=args.pretrain_task, feature_type=args.feature_type)
        if args.pretrain_task == 'DGI':
            pt = DGI(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device, pretrain_task=args.pretrain_task, feature_type=args.feature_type)
        if args.pretrain_task == 'NodeMultiGprompt':
            nonlinearity = 'prelu'
            pt = NodePrePrompt(args.dataset_name, args.hid_dim, nonlinearity, 0.9, 0.9, 0.1, 0.001, 1, 0.3)
        if args.pretrain_task == 'GraphMultiGprompt':
            nonlinearity = 'prelu'
            pt = GraphPrePrompt(graph_list, input_dim, out_dim, args.dataset_name, args.hid_dim, nonlinearity,0.9,0.9,0.1,1,0.3)
        if args.pretrain_task == 'GraphMAE':
            pt = GraphMAE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device,
                        mask_rate=0.75, drop_edge_rate=0.0, replace_rate=0.1, loss_fn='sce', alpha_l=2, pretrain_task=args.pretrain_task, feature_type=args.feature_type)
        pt.pretrain()
