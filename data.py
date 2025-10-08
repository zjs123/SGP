import torch
import pickle as pk
from random import shuffle
import random
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor, WikipediaNetwork
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data,Batch
from torch_geometric.utils import negative_sampling
from torch_geometric.loader.cluster import ClusterData
import os
import numpy

from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

# from collections import defaultdict
import pickle as pk
from torch_geometric.utils import subgraph, k_hop_subgraph
import torch
import numpy as np
from torch_geometric.data import Data, Batch
import random
import os
from random import shuffle
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_geometric.data import Data
import numpy as np
import pickle
from tqdm import tqdm

class GraphDataset(Dataset):
    def __init__(self, graphs):
        """
        初始化 GraphDataset
        :param graphs: 包含图对象的列表
        """
        super(GraphDataset, self).__init__()
        self.graphs = graphs

    def len(self):
        """
        返回数据集的大小
        :return: 数据集的大小
        """
        return len(self.graphs)

    def get(self, idx):
        """
        获取索引为 idx 的图
        :param idx: 索引
        :return: 图对象
        """
        graph = self.graphs[idx]
        # 可以在这里进行图数据的预处理或特征提取
        # 例如，如果每个图对象都有节点特征和边特征，可以返回它们
        # return {'node_features': graph.node_features, 'edge_index': graph.edge_index}
        return graph


def split_induced_graphs(data, dir_path, device, idx_train, idx_test, smallest_size=10, largest_size=30, task_type = 'node', feature_type = 'None'):
    if task_type == 'node':
        graph_list = []
        from copy import deepcopy
        
        for id_list in [idx_train, idx_test]:
            sub_graph_list = []
            for index in tqdm(id_list):
                index = int(index)
                current_label = data.y[index].item()

                subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=1,
                                                    edge_index=data.edge_index, relabel_nodes=False)

                subset = subset.to(device)
                sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
                sub_edge_index = sub_edge_index.to(device)

                x = data.x[subset]

                induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label, index = index)
                sub_graph_list.append(deepcopy(induced_graph).to('cpu'))
            graph_list.append(sub_graph_list)


        if not os.path.exists(dir_path):
            os.makedirs(dir_path) 

        file_path = os.path.join(dir_path, 'induced_graph_min'+ str(smallest_size) +'_max'+str(largest_size)+'_'+str(feature_type)+'.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(graph_list, f)
            print("node induced graph data has been write into " + file_path)
    
    if task_type == 'edge':
        graph_list = []
        from copy import deepcopy
        
        for id_list in [idx_train, idx_test]:
            sub_graph_list = []
            for index in tqdm(id_list):
                index = int(index)
                source_index, target_index = int(data.edge_index[0][index]), int(data.edge_index[1][index])
                current_label = data.edge_y[index].item()

                subset_source, _, _, _ = k_hop_subgraph(node_idx=source_index, num_hops=1,
                                                    edge_index=data.edge_index, relabel_nodes=False)

                subset_target, _, _, _ = k_hop_subgraph(node_idx=target_index, num_hops=1,
                                                    edge_index=data.edge_index, relabel_nodes=False)

                subset = torch.cat([subset_source, subset_target], dim = 0).to(device)
                sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
                sub_edge_index = sub_edge_index.to(device)

                x = data.x[subset]

                induced_graph = Data(x=x, edge_index=sub_edge_index, edge_y=current_label, index = index)
                sub_graph_list.append(deepcopy(induced_graph).to('cpu'))
            graph_list.append(sub_graph_list)


        if not os.path.exists(dir_path):
            os.makedirs(dir_path) 

        file_path = os.path.join(dir_path, 'induced_graph_min'+ str(smallest_size) +'_max'+str(largest_size)+'_'+str(feature_type)+'.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(graph_list, f)
            print("edge induced graph data has been write into " + file_path)
'''
def node_sample_and_save(data, k, folder, num_classes):
    # 获取标签
    labels = data.y.to('cpu')
    useable_idx = torch.arange(start=0, end=len(labels), step=1)[labels != -1]
    random_idx = torch.randperm(len(useable_idx))
    
    if k <= 100: # few_shot
        # 随机选择90%的数据作为测试集
        num_test = int(0.9 * len(useable_idx))
        if num_test < 1000:
            num_test = int(0.7 * len(useable_idx))
        test_idx = useable_idx[:num_test]
        test_labels = labels[test_idx]
        
        # 剩下的作为候选训练集
        remaining_idx = useable_idx[num_test:]
        remaining_labels = labels[remaining_idx]
        
        # 从剩下的数据中选出k*标签数个样本作为训练集
        if num_classes >= 2:
            train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(num_classes)])
        else:
            train_idx = remaining_idx[:k]
        print([k, num_classes, len(train_idx)])
        shuffled_indices = torch.randperm(train_idx.size(0))
        train_idx = train_idx[shuffled_indices]
        train_labels = labels[train_idx]
    else:
        num_test = int(0.1 * len(useable_idx))
        test_idx = useable_idx[:num_test]
        test_labels = labels[test_idx]
        
        # 剩下的作为训练集
        remaining_idx = useable_idx[num_test:]
        remaining_labels = labels[remaining_idx]
        
        # 剩下的数据作为训练集
        shuffled_indices = torch.randperm(remaining_idx.size(0))
        train_idx = remaining_idx[shuffled_indices]
        train_labels = labels[train_idx]

    # 保存文件
    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

def edge_sample_and_save(data, k, folder, num_classes):
    # 获取标签
    labels = data.edge_y.to('cpu')
    useable_idx = torch.arange(start=0, end=len(labels), step=1)[labels != -1]
    random_idx = torch.randperm(len(useable_idx))

    # 选出k*标签数个样本作为训练集
    if num_classes >= 2:
        train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(num_classes)])
    else:
        train_idx = remaining_idx[:k]
    print([k, num_classes, len(train_idx)])
    shuffled_indices = torch.randperm(train_idx.size(0))
    train_idx = train_idx[shuffled_indices]
    train_labels = labels[train_idx]
    
    if k <= 100: # few_shot
        # 随机选择90%的数据作为测试集
        num_test = int(0.9 * len(useable_idx))
        if num_test < 1000:
            num_test = int(0.7 * len(useable_idx))
        test_idx = useable_idx[random_idx[:num_test]]
        test_labels = labels[test_idx]
        
        # 剩下的作为候选训练集
        remaining_idx = useable_idx[random_idx[num_test:]]
        remaining_labels = labels[remaining_idx]
        
        # 选出k*标签数个样本作为训练集
        if num_classes >= 2:
            train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(num_classes)])
        else:
            train_idx = remaining_idx[:k]
        print([k, num_classes, len(train_idx)])
        shuffled_indices = torch.randperm(train_idx.size(0))
        train_idx = train_idx[shuffled_indices]
        train_labels = labels[train_idx]
    else:
        num_test = int(0.1 * len(useable_idx))
        test_idx = useable_idx[random_idx[:num_test]]
        test_labels = labels[test_idx]
        
        # 剩下的作为训练集
        remaining_idx = useable_idx[random_idx[num_test:]]
        remaining_labels = labels[remaining_idx]
        
        # 剩下的数据作为训练集
        shuffled_indices = torch.randperm(remaining_idx.size(0))
        train_idx = remaining_idx[shuffled_indices]
        train_labels = labels[train_idx]

    # 保存文件
    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))
'''

def node_sample_and_save(data, k, folder, num_classes):
    # 获取标签+打乱数据
    labels = data.y.to('cpu')
    useable_idx = torch.arange(start=0, end=len(labels), step=1)[labels != -1]
    random_idx = torch.randperm(len(useable_idx))
    useable_idx = useable_idx[random_idx]
    useable_labels = labels[useable_idx]
    
    if k <= 10: # few_shot
        # 选出k*标签数个样本作为训练集
        if num_classes >= 2:
            train_idx = torch.cat([useable_idx[useable_labels == i][:k] for i in range(num_classes)])
        else:
            train_idx = useable_idx[:k]
        print([k, num_classes, len(train_idx)])
        shuffled_indices = torch.randperm(train_idx.size(0))
        train_idx = train_idx[shuffled_indices]
        train_labels = labels[train_idx]

        # 剩下的每个类别随机选择1000个样本作为测试集
        remaining_idx = useable_idx[~np.isin(useable_idx, train_idx)]
        remaining_labels = useable_labels[~np.isin(useable_idx, train_idx)]
        if num_classes >= 2:
            test_idx = torch.cat([remaining_idx[remaining_labels == i][:1000] for i in range(num_classes)])
        else:
            test_idx = remaining_idx[:1000]
        print([num_classes, len(test_idx)])
        shuffled_indices = torch.randperm(test_idx.size(0))
        test_idx = test_idx[shuffled_indices]
        test_labels = labels[test_idx]
    else:
        num_test = int(0.2 * len(useable_idx))
        test_idx = useable_idx[:num_test]
        test_labels = labels[test_idx]
        
        # 剩下的作为训练集
        remaining_idx = useable_idx[num_test:]
        remaining_labels = labels[remaining_idx]
        
        # 剩下的数据作为训练集
        shuffled_indices = torch.randperm(remaining_idx.size(0))
        train_idx = remaining_idx[shuffled_indices]
        train_labels = labels[train_idx]

    # 保存文件
    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

def edge_sample_and_save(data, k, folder, num_classes):
    # 获取标签+打乱数据
    labels = data.edge_y.to('cpu')
    useable_idx = torch.arange(start=0, end=len(labels), step=1)[labels != -1]
    random_idx = torch.randperm(len(useable_idx))
    useable_idx = useable_idx[random_idx]
    useable_labels = labels[useable_idx]

    if k <= 100: # few_shot
        # 选出k*标签数个样本作为训练集
        if num_classes >= 2:
            train_idx = torch.cat([useable_idx[useable_labels == i][:k] for i in range(num_classes)])
        else:
            train_idx = useable_idx[:k]
        print([k, num_classes, len(train_idx)])
        shuffled_indices = torch.randperm(train_idx.size(0))
        train_idx = train_idx[shuffled_indices]
        train_labels = labels[train_idx]

        # 剩下的每个类别随机选择1000个样本作为测试集
        remaining_idx = useable_idx[~np.isin(useable_idx, train_idx)]
        remaining_labels = useable_labels[~np.isin(useable_idx, train_idx)]
        if num_classes >= 2:
            test_idx = torch.cat([remaining_idx[remaining_labels == i][:1000] for i in range(num_classes)])
        else:
            test_idx = remaining_idx[:1000]
        print([num_classes, len(test_idx)])
        shuffled_indices = torch.randperm(test_idx.size(0))
        test_idx = test_idx[shuffled_indices]
        test_labels = labels[test_idx]
    else:
        num_test = int(0.2 * len(useable_idx))
        test_idx = useable_idx[:num_test]
        test_labels = labels[test_idx]
        
        # 剩下的作为训练集
        remaining_idx = useable_idx[num_test:]
        remaining_labels = labels[remaining_idx]
        
        # 剩下的数据作为训练集
        shuffled_indices = torch.randperm(remaining_idx.size(0))
        train_idx = remaining_idx[shuffled_indices]
        train_labels = labels[train_idx]


    # 保存文件
    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))


def graph_sample_and_save(dataset, k, folder, num_classes):

    # 计算测试集的数量（例如80%的图作为测试集）
    num_graphs = len(dataset)
    num_test = int(0.8 * num_graphs)

    labels = torch.tensor([graph.y.item() for graph in dataset])

    # 随机选择测试集的图索引
    all_indices = torch.randperm(num_graphs)
    test_indices = all_indices[:num_test]
    torch.save(test_indices, os.path.join(folder, 'test_idx.pt'))
    test_labels = labels[test_indices]
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))

    remaining_indices = all_indices[num_test:]

    # 从剩下的10%的图中为训练集选择每个类别的k个样本
    train_indices = []
    for i in range(num_classes):
        # 选出该类别的所有图
        class_indices = [idx for idx in remaining_indices if labels[idx].item() == i]
        # 如果选出的图少于k个，就取所有该类的图
        selected_indices = class_indices[:k] 
        train_indices.extend(selected_indices)
    print([k, num_classes, len(train_indices)])

    # 随机打乱训练集的图索引
    train_indices = torch.tensor(train_indices)
    shuffled_indices = torch.randperm(train_indices.size(0))
    train_indices = train_indices[shuffled_indices]
    torch.save(train_indices, os.path.join(folder, 'train_idx.pt'))
    train_labels = labels[train_indices]
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))

def node_degree_as_features(data_list):
    from torch_geometric.utils import degree
    for data in data_list:
        # 计算所有节点的度数，这将返回一个张量
        deg = degree(data.edge_index[0], dtype=torch.long)

        # 将度数张量变形为[nodes, 1]以便与其他特征拼接
        deg = deg.view(-1, 1).float()
        
        # 如果原始数据没有节点特征，可以直接使用度数作为特征
        if data.x is None:
            data.x = deg
        else:
            # 将度数特征拼接到现有的节点特征上
            data.x = torch.cat([data.x, deg], dim=1)


def load4graph(dataname, feature_type = 'None', pretrained=False):
    r"""A plain old python object modeling a batch of graphs as one big
        (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
        base class, all its methods can also be used here.
        In addition, single graphs can be reconstructed via the assignment vector
        :obj:`batch`, which maps each node to its respective graph identifier.
        """

    if dataname in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD']:
        dataset = TUDataset(root='data/TUDataset', name=dataname, use_node_attr=True)  # use_node_attr=False时，节点属性为one-hot编码的节点类别
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        if dataname in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
            graph_list = [g for g in graph_list]
            node_degree_as_features(graph_list)
            input_dim = graph_list[0].x.size(1)   
        
    if dataname in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataname, root='./dataset')
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

    if pretrained == False:
        for data in dataset:  
            # use random node feature
            if feature_type == 'random':
                random_node_f = torch.Tensor(data.x.size())
                torch.nn.init.xavier_uniform_(random_node_f, gain=1)
                data.x = random_node_f
            
            # use text node feature
            if feature_type == 'text':
                pass
                
            # use structure node feature
            if feature_type == 'structure':
                pe = T.AddLaplacianEigenvectorPE(k = 256, attr_name = 'x', is_undirected = True) #T.AddRandomWalkPE(walk_length = 5, attr_name = 'x')
                data = pe(data)
                data.x = F.normalize(data.x, 2, -1)
                input_dim = len(data.x[0])

        return dataset, input_dim, out_dim
    
    else:
        for data in graph_list:  
            # use random node feature
            if feature_type == 'random':
                random_node_f = torch.Tensor(data.x.size())
                torch.nn.init.xavier_uniform_(random_node_f, gain=1)
                data.x = random_node_f
            
            # use text node feature
            if feature_type == 'text':
                pass
                
            # use structure node feature
            if feature_type == 'structure':
                pe = T.AddLaplacianEigenvectorPE(k = 256, attr_name = 'x', is_undirected = True) #T.AddRandomWalkPE(walk_length = 5, attr_name = 'x')
                data = pe(data)
                data.x = F.normalize(data.x, 2, -1)
                input_dim = len(data.x[0])

        return graph_list, input_dim, out_dim

    
def load4node_edge(dataname, feature_type = 'None', task_name = 'Pre_train'):
    print(dataname)
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        dataset = Planetoid(root='datasets/Planetoid', name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['chameleon', 'crocodile', 'squirrel']:
        dataset = WikipediaNetwork(root='datasets/Wikipedia', name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Computers', 'Photo']:
        dataset = Amazon(root='data/amazon', name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Flickr':
        dataset = Flickr(root='data/Flickr')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname in ['Wisconsin', 'Texas', 'Cornell']:
        dataset = WebKB(root='data/'+dataname, name=dataname)
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'Actor':
        dataset = Actor(root='data/Actor')
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    elif dataname == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./dataset')
        data = dataset[0]
        input_dim = data.x.shape[1]
        out_dim = dataset.num_classes
    elif dataname in ['ENZYMES', 'PROTEINS']:
        # 实现TUDataset中两个multi graphs dataset的节点分类
        dataset = TUDataset(root='data/TUDataset', name=dataname, use_node_attr=True)
        node_class = dataset.data.x[:,-3:]
        input_dim = dataset.num_node_features
        out_dim = dataset.num_node_labels
        data = Batch.from_data_list(dataset)  # 将dataset中小图合并成一个大图
        data.y = node_class.nonzero().T[1]
    
    # use random node feature
    if feature_type == 'random':
        random_node_f = torch.Tensor(data.x.size())
        torch.nn.init.xavier_uniform_(random_node_f, gain=1)
        data.x = random_node_f
    
    # use text node feature
    if feature_type == 'text':
        pass
        
    # use structure node feature
    if feature_type == 'structure':
        pe = T.AddLaplacianEigenvectorPE(k = 256, attr_name = 'x', is_undirected = True) #T.AddRandomWalkPE(walk_length = 5, attr_name = 'x')
        data = pe(data)
        data.x = F.normalize(data.x, 2, -1)
        input_dim = len(data.x[0])

    return data, input_dim, out_dim

# used in pre_train.py
def NodePretrain(data, num_parts=200, split_method='Random Walk'):

    # if(dataname=='Cora'):
    #     num_parts=220
    # elif(dataname=='Texas'):
    #     num_parts=20
    if(split_method=='Cluster'):
        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        
        graph_list = list(ClusterData(data=data, num_parts=num_parts))
    elif(split_method=='Random Walk'):
        from torch_cluster import random_walk
        split_ratio = 0.1
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
        walk_list = random_walk(data.edge_index[0], data.edge_index[1], random_node_list, walk_length=walk_length)

        graph_list = [] 
        skip_num = 0        
        for walk in walk_list:   
            subgraph_nodes = torch.unique(walk)
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)

            graph_list.append(subgraph_data)

        print(f"Total {len(graph_list)} random walk subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")

    else:
        print('None split method!')
        exit()
    
    return graph_list