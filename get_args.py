import argparse

def get_pretrain_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--pretrain_task', type = str, default='GraphMAE') # LP DGI GraphCL GraphMAE
    parser.add_argument('--feature_type', type = str, default='text') # text random structure
    parser.add_argument('--dataset_name', type=str, default='Cornell',help='Choose the dataset of pretrainor downstream task')
    parser.add_argument('--device', type=int, default=0, # Semantic_Scholar
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_type', type=str, default="GCN", help='We support gnn like \GCN\ \GAT\ \GraphTransformer\ \GCov\ \GIN\ \GraphSAGE\, please read ProG.model module')
    parser.add_argument('--hid_dim', type=int, default=128,
                        help='hideen layer of GNN dimensions (default: 128)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='Number of GNN message passing layers (default: 2).')

    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='How the node features across layers are combined. last, sum, max or concat')

    parser.add_argument('--seed', type=int, default= 42, help = "Seed for splitting dataset.")
    parser.add_argument('--runseed', type=int, default= 0, help = "Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default = 0, help='Number of workers for dataset loading')
    parser.add_argument('--num_layers', type=int, default = 1, help='A range of [1,2,3]-layer MLPs with equal width')

    args = parser.parse_args()
    return args

def get_task_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of fine-tuning of graph neural networks')
    parser.add_argument('--downstream_task', type = str, default='None') #  sector_cat influencial_cat popular_category avg_score cite_num strength_category
    parser.add_argument('--shot_num', type=int, default = 1, help='Number of shots') # 1, 3, 5, 10, 1000
    parser.add_argument('--device', type=int, default = 0, help='Which gpu to use if any (default: 0)')
    parser.add_argument('--task_train_type', type=str, default="label_prompt", # All-in-one
                        help='Choose the prompt type for node or graph task, for node task,we support \GPPT\, \All-in-one\, \Gprompt\ for graph task , \All-in-one\, \Gprompt\, \GPF\, \GPF-plus\ ')
    parser.add_argument('--batch_size', type=int, default = 128,
                        help='Input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default = 2000,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--pre_train_model_path', type=str, default= '/home/jz875/project/label_prompt/pre_trained_model/COX2/text.GraphMAE.GCN.128.2.pth', #'/home/jz875/project/label_prompt/pre_trained_model/ICEWS/text.LP.GCN.128.2.pth', #/home/jz875/project/label_prompt/pre_trained_model/ICEWS/text.LP.GCN.128hidden_dim.pth #/home/jz875/project/label_prompt/pre_trained_model/Amazon/text.LP.GCN.128hidden_dim.pth 
                        help='add pre_train_model_path to the downstream task, the model is self-supervise model if the path is None and prompttype is None.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')

    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default = 0, help='Number of workers for dataset loading')
    parser.add_argument('--pnum', type=int, default = 5, help='The number of independent basis for GPF-plus')
    parser.add_argument('--task_num', type=int, default = 5, help='The number of tasks for computing the mean metrices')

    args = parser.parse_args()
    return args

    '''
    1 shot full train structure f
    ['Cora', 'sector_cat', 1, 7, '']
    ['task_prompt', ['MLP', 'MLP', 'MLP', 'MLP', 'MLP']]
    ['ACC', 0.3011, 0.0423]
    ['Macro_F1', 0.2834, 0.0489]
    ['Micro_F1', 0.3011, 0.0423]
    ['Weighted_F1', 0.2919, 0.0525]
    ['ROC', 0.6669, 0.0457]
    ['PRC', 0.3589, 0.0492]

    ['Cora', 'sector_cat', 1, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/structure.GraphMAE.GCN.128.2.pth']
    ['task_prompt', ['GPPT', 'GPPT', 'GPPT', 'GPPT', 'GPPT']]
    ['ACC', 0.2985, 0.0457]
    ['Macro_F1', 0.278, 0.0542]
    ['Micro_F1', 0.2985, 0.0457]
    ['Weighted_F1', 0.2928, 0.0537]
    ['ROC', 0.664, 0.0618]
    ['PRC', 0.3601, 0.0633]

    ['Cora', 'sector_cat', 1, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/structure.GraphMAE.GCN.128.2.pth']
    ['task_prompt', ['GPF', 'GPF', 'GPF', 'GPF', 'GPF']]
    ['ACC', 0.2843, 0.0492]
    ['Macro_F1', 0.2611, 0.0375]
    ['Micro_F1', 0.2843, 0.0492]
    ['Weighted_F1', 0.2777, 0.0394]
    ['ROC', 0.6346, 0.0391]
    ['PRC', 0.3125, 0.0415]

    'Cora', 'sector_cat', 1, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/structure.GraphMAE.GCN.128.2.pth']
    ['task_prompt', ['All in one', 'All in one', 'All in one', 'All in one', 'All in one']]
    ['ACC', 0.1986, 0.0893]
    ['Macro_F1', 0.1451, 0.0606]
    ['Micro_F1', 0.1986, 0.0893]
    ['Weighted_F1', 0.1441, 0.082]
    ['ROC', 0.6229, 0.0283]
    ['PRC', 0.2761, 0.028]

    ['Cora', 'sector_cat', 1, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/structure.GraphMAE.GCN.128.2.pth']
    ['task_prompt', ['Gprompt', 'Gprompt', 'Gprompt', 'Gprompt', 'Gprompt']]
    ['ACC', 0.2438, 0.0368]
    ['Macro_F1', 0.2341, 0.0402]
    ['Micro_F1', 0.2438, 0.0368]
    ['Weighted_F1', 0.244, 0.0441]
    ['ROC', 0.6117, 0.0403]
    ['PRC', 0.2845, 0.0385]

    ['Cora', 'sector_cat', 1, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/structure.GraphMAE.GCN.128.2.pth']
    ['task_prompt', ['label', 'label', 'label', 'label', 'label']]
    ['ACC', 0.3385, 0.0458]
    ['Macro_F1', 0.3167, 0.0422]
    ['Micro_F1', 0.3385, 0.0458]
    ['Weighted_F1', 0.3373, 0.0389]
    ['ROC', 0.6991, 0.0447]
    ['PRC', 0.3917, 0.0447]
    '''

    '''
    1 shot full train text f
    ['ACC', 0.3025, 0.0704]
    ['Macro_F1', 0.2633, 0.072]
    ['Micro_F1', 0.3025, 0.0704]
    ['Weighted_F1', 0.2747, 0.0852]
    ['ROC', 0.7663, 0.0485]
    ['PRC', 0.471, 0.0598]

    ['Cora', 'sector_cat', 1, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/text.GraphMAE.GCN.128.2.pth']
    ['task_prompt', ['All in one', 'All in one', 'All in one', 'All in one', 'All in one']]
    ['ACC', 0.3572, 0.0948]
    ['Macro_F1', 0.2733, 0.0603]
    ['Micro_F1', 0.3572, 0.0948]
    ['Weighted_F1', 0.2811, 0.097]
    ['ROC', 0.7815, 0.0301]
    ['PRC', 0.4543, 0.0614]

    ['Cora', 'sector_cat', 1, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/text.GraphMAE.GCN.128.2.pth']
    ['task_prompt', ['GPF', 'GPF', 'GPF', 'GPF', 'GPF']]
    ['ACC', 0.4874, 0.0715]
    ['Macro_F1', 0.4707, 0.0819]
    ['Micro_F1', 0.4874, 0.0715]
    ['Weighted_F1', 0.4771, 0.0832]
    ['ROC', 0.8414, 0.0432]
    ['PRC', 0.5688, 0.0868]

    ['Cora', 'sector_cat', 1, 7, '/home/jz875/project/label_prompt/pre_trained_model/Cora/text.GraphMAE.GCN.128.2.pth']
    ['task_prompt', ['label', 'label', 'label', 'label', 'label']]
    ['ACC', 0.6195, 0.0745]
    ['Macro_F1', 0.6073, 0.0599]
    ['Micro_F1', 0.6195, 0.0745]
    ['Weighted_F1', 0.6204, 0.0861]
    ['ROC', 0.897, 0.031]
    ['PRC', 0.6946, 0.0588]
    '''

    '''
    ['COX2', 'None', 5, 2, '/home/jz875/project/label_prompt/pre_trained_model/COX2/text.GraphMAE.GCN.128.2.pth']
    ['task_prompt', ['label_prompt', 'label_prompt', 'label_prompt', 'label_prompt', 'label_prompt']]
    ['ACC', 0.5764, 0.0618]
    ['Macro_F1', 0.5237, 0.0442]
    ['Micro_F1', 0.5764, 0.0618]
    ['Weighted_F1', 0.6112, 0.0545]
    ['ROC', 0.5964, 0.0594]
    ['PRC', 0.5652, 0.0387]
    '''