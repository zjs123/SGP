import torch
from torch_geometric.loader import DataLoader
from task_strategies import base
import time
import warnings
import numpy as np
from data import load4node_edge, load4graph, graph_sample_and_save, split_induced_graphs, GraphDataset
import pickle
import os
warnings.filterwarnings("ignore")

import torchmetrics
from torch_geometric.data import Batch
import torch
import random
from tqdm import tqdm
import torch.nn.functional as F
from get_args import get_pretrain_args, get_task_args

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

def constraint(device,prompt):
    if isinstance(prompt,list):
        sum=0
        for p in prompt:
            sum=sum+torch.norm(torch.mm(p,p.T)-torch.eye(p.shape[0]).to(device))
        return sum/len(prompt)
    else:
        return torch.norm(torch.mm(prompt,prompt.T)-torch.eye(prompt.shape[0]).to(device))

def center_embedding(input_, index, label_num):
    device=input_.device
    c = torch.zeros(label_num, input_.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index.unsqueeze(1).expand(-1, input_.size(1)), src=input_)
    class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input_.dtype, device=device)

    # Take the average embeddings for each class
    # If directly divided the variable 'c', maybe encountering zero values in 'class_counts', such as the class_counts=[[0.],[4.]]
    # So we need to judge every value in 'class_counts' one by one, and seperately divided them.
    # output_c = c/class_counts
    for i in range(label_num):
        if(class_counts[i].item()==0):
            continue
        else:
            c[i] /= class_counts[i]

    return c, class_counts

class GraphTask(base.BaseTask):
    def __init__(self, data, input_dim, output_dim, task_num = 5 , *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.data = data
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.task_type = 'GraphTask'
        self.task_num = task_num
        self.feature_type = feature_type

        if self.shot_num > 0:
            self.create_few_data_folder()

    def create_few_data_folder(self):
        # 创建文件夹并保存数据
        for k in [1, 3, 5, 10, 50, 100]:
            k_shot_folder = 'datasets/few_shot_data/'+self.dataset_name+'/'+ self.downstream_task +'/' + str(k) +'_shot'
            os.makedirs(k_shot_folder, exist_ok=True)
            for i in range(1, 6):
                folder = os.path.join(k_shot_folder, str(i))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    graph_sample_and_save(self.data, k, folder, self.output_dim)
                    print(str(k) + ' shot ' + str(i) + ' th is saved!!')

    def node_degree_as_features(self, data_list):
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

    def MLP_finetune_train(self, train_loader):
        self.gnn.train()
        self.answering.train()
        total_loss = 0.0
        for batch in train_loader:  
            self.optimizer.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)  

            
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)

    def label_prompt_train(self, train_loader):
        self.prompt.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        for batch in train_loader:  
            self.optimizer.zero_grad() 
            batch = batch.to(self.device)
            critical_emb = self.prompt.get_critical_emb(self.gnn, batch.x, batch.edge_index, batch.batch, is_train = True)
            out = self.prompt.answering(critical_emb)
            loss = self.criterion(out, batch.y)
            max_loss, min_loss, noise_loss = self.prompt.MI_loss(self.gnn, batch)

            loss += max_loss + noise_loss + min_loss
            
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)
        
    def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
        #we update answering and prompt alternately.
        
        # tune task head
        self.answering.train()
        self.prompt.eval()
        for epoch in range(1, answer_epoch + 1):
            answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
            print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

        # tune prompt
        self.answering.eval()
        self.prompt.train()
        for epoch in range(1, prompt_epoch + 1):
            pg_loss = self.prompt.Tune( train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
            print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
        
        return pg_loss

    def GPFTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0 
        for batch in train_loader:  
            self.optimizer.zero_grad() 
            batch = batch.to(self.device)
            batch.x = self.prompt.add(batch.x)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, task_train_type = self.task_train_type)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)  
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)  

    def GpromptTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0
        accumulated_centers = None
        accumulated_counts = None

        for batch in train_loader:
            # archived code for complete prototype embeddings of each labels. Not as well as batch version
            # # compute the prototype embeddings of each type of label
            self.pg_opi.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, task_train_type = 'Gprompt')
            # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
            center, class_counts = center_embedding(out,batch.y, self.output_dim)
            # 累积中心向量和样本数
            if accumulated_centers is None:
                accumulated_centers = center
                accumulated_counts = class_counts
            else:
                accumulated_centers += center * class_counts
                accumulated_counts += class_counts
            loss = self.criterion(out, center, batch.y)
            loss.backward()  
            self.pg_opi.step()  
            total_loss += loss.item()
            # 计算加权平均中心向量
            mean_centers = accumulated_centers / accumulated_counts

            return total_loss / len(train_loader), mean_centers

    def GPPTtrain(self, train_loader):
        self.prompt.train()
        for batch in train_loader:
            temp_loss=torch.tensor(0.0,requires_grad=True).to(self.device)
            graph_list = batch.to_data_list()        
            for index, graph in enumerate(graph_list):
                graph=graph.to(self.device)              
                node_embedding = self.gnn(graph.x, graph.edge_index)
                out = self.prompt(node_embedding, graph.edge_index) # gppt下游在1-shot的时候，prompt结果为nan
                loss = self.criterion(out, torch.full((1,graph.x.shape[0]), graph.y.item()).reshape(-1).to(self.device))
                temp_loss += loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())           
            temp_loss = temp_loss/(index+1)
            self.pg_opi.zero_grad()
            temp_loss.backward()
            self.pg_opi.step()
            self.prompt.update_StructureToken_weight(self.prompt.get_mid_h())
        return temp_loss.item()

    def run(self, f):
        test_metrics = dict()
        test_accs = []
        f1s = []
        rocs = []
        prcs = []
        batch_best_loss = []
        if self.task_train_type == 'All-in-one':
            self.answer_epoch = 1
            self.prompt_epoch = 1
            self.epochs = int(self.epochs/self.answer_epoch)
        for i in range(1, 6):
            
            self.initialize_gnn()
            self.initialize_prompt()
            self.answering = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim)).to(self.device)
            self.initialize_optimizer()

            idx_train = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i)).type(torch.long).to(self.device)
            print('idx_train',idx_train)
            train_lbls = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
            print("true",i,train_lbls)
            
            idx_test = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i)).type(torch.long).to(self.device)
            test_lbls = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
        
            train_dataset = self.data[idx_train]
            test_dataset = self.data[idx_test]
            #print(train_dataset[0])

            if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa']:
                train_dataset = [train_g for train_g in train_dataset]
                test_dataset = [test_g for test_g in test_dataset]
                self.node_degree_as_features(train_dataset)
                self.node_degree_as_features(test_dataset)
                if self.task_train_type == 'GPPT':
                    processed_dataset = [g for g in self.dataset]
                    self.node_degree_as_features(processed_dataset)
                    processed_dataset = Batch.from_data_list([g for g in processed_dataset])
                self.input_dim = train_dataset[0].x.size(1)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            print("prepare data is finished!")

            patience = 50
            best = 1e9
            cnt_wait = 0
            
            if self.task_train_type == 'GPPT':
                # initialize the GPPT hyperparametes via graph data
                if self.dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY', 'ogbg-ppa']:

                    total_num_nodes = sum([data.num_nodes for data in train_dataset])
                    train_node_ids = torch.arange(0,total_num_nodes).squeeze().to(self.device)
                    self.gppt_loader = DataLoader(processed_dataset.to_data_list(), batch_size=1, shuffle=False)
                    for i, batch in enumerate(self.gppt_loader):
                        if(i==0):
                            node_for_graph_labels = torch.full((1,batch.x.shape[0]), batch.y.item())
                            node_embedding = self.gnn(batch.x.to(self.device), batch.edge_index.to(self.device))
                        else:                   
                            node_for_graph_labels = torch.concat([node_for_graph_labels,torch.full((1,batch.x.shape[0]), batch.y.item())],dim=1)
                            node_embedding = torch.concat([node_embedding,self.gnn(batch.x.to(self.device), batch.edge_index.to(self.device))],dim=0)
                    
                    node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)
                    self.prompt.weigth_init(node_embedding,processed_dataset.edge_index.to(self.device), node_for_graph_labels, train_node_ids)

                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)                    
                else:
                    train_node_ids = torch.arange(0,train_dataset.x.shape[0]).squeeze().to(self.device)
                    # 将子图的节点id转换为全图的节点id
                    iterate_id_num = 0
                    for index, g in enumerate(train_dataset):
                        current_node_ids = iterate_id_num+torch.arange(0,g.x.shape[0]).squeeze().to(self.device)
                        iterate_id_num += g.x.shape[0]
                        previous_node_num = sum([self.data[i].x.shape[0] for i in range(idx_train[index]-1)])
                        train_node_ids[current_node_ids] += previous_node_num

                    self.gppt_loader = DataLoader(self.data, batch_size=1, shuffle=True)
                    for i, batch in enumerate(self.gppt_loader):
                        if(i==0):
                            node_for_graph_labels = torch.full((1,batch.x.shape[0]), batch.y.item())
                        else:                   
                            node_for_graph_labels = torch.concat([node_for_graph_labels, torch.full((1,batch.x.shape[0]), batch.y.item())],dim=1)
                    
                    node_embedding = self.gnn(self.data.x.to(self.device), self.data.edge_index.to(self.device))
                    node_for_graph_labels=node_for_graph_labels.reshape((-1)).to(self.device)
                    
                    self.prompt.weigth_init(node_embedding, self.data.edge_index.to(self.device), node_for_graph_labels, train_node_ids)
                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()
                center = None
                if self.task_train_type == 'MLP_finetune':
                    loss = self.MLP_finetune_train(train_loader)
                elif self.task_train_type == 'label_prompt':  
                    #self.prompt.get_label_key_emb(self.gnn, train_dataset, idx_train)
                    loss = self.label_prompt_train(train_loader)
                elif self.task_train_type == 'All-in-one':
                    loss = self.AllInOneTrain(train_loader, self.answer_epoch, self.prompt_epoch)
                elif self.task_train_type in ['GPF', 'GPF-plus']:
                    loss = self.GPFTrain(train_loader)
                elif self.task_train_type =='Gprompt':
                    loss, center = self.GpromptTrain(train_loader)
                elif self.task_train_type =='GPPT':
                    loss = self.GPPTtrain(train_loader)
                        
                if loss < best:
                    best = loss
                    # best_t = epoch
                    cnt_wait = 0
                    # torch.save(model.state_dict(), args.save_name)
                else:
                    cnt_wait += 1
                    if cnt_wait == patience:
                            print('-' * 100)
                            print('Early stopping at '+str(epoch) +' eopch!')
                            break
                print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))
                
            import math
            if not math.isnan(loss):
                batch_best_loss.append(loss)
                if self.task_train_type == 'label_prompt': 
                    metric_dict = self.prompt.Eva_Graph(test_loader, self.data, idx_test, self.gnn, None, self.output_dim, self.device, center)
                else:
                    metric_dict = self.prompt.Eva_Graph(test_loader, self.data, idx_test, self.gnn, self.answering, self.output_dim, self.device, center)
                if len(test_metrics.keys()) == 0:
                    for metric in metric_dict.keys():
                        test_metrics[metric] = []
                        test_metrics[metric].append(metric_dict[metric])
                else:
                    for metric in metric_dict.keys():
                        test_metrics[metric].append(metric_dict[metric])
    
        print([self.dataset_name, self.downstream_task, self.shot_num, self.output_dim, self.pre_train_model_path])
        f.write(str([self.dataset_name, self.downstream_task, self.shot_num, self.output_dim, self.pre_train_model_path])+'\n')
        for metric_name in test_metrics.keys():
            if metric_name == 'task_prompt':
                print([metric_name, test_metrics[metric_name]])
                f.write(str([metric_name, test_metrics[metric_name]])+'\n')
            else:
                print([metric_name, np.around(np.mean(test_metrics[metric_name]),4), np.around(np.std(test_metrics[metric_name]),4)])
                f.write(str([metric_name, np.around(np.mean(test_metrics[metric_name]),4), np.around(np.std(test_metrics[metric_name]),4)])+'\n')

        print(self.pre_train_type, self.gnn_type, self.task_train_type, self.downstream_task, "Task completed")
        f.write(str([self.pre_train_type, self.gnn_type, self.task_train_type, self.downstream_task, "Task completed"])+'\n')
        f.write('\n')
        mean_best = np.mean(batch_best_loss)

        return  test_metrics

# start training
if __name__ == '__main__':
    args = get_task_args()
    shot_num = args.shot_num
    task_train_type = args.task_train_type
    downstream_task = args.downstream_task
    pre_train_model_path = args.pre_train_model_path
    print('shot_num: ', shot_num)
    print('task_train_type: ', task_train_type)
    print('pre_trained_model: ', pre_train_model_path)

    if '.pth' not in pre_train_model_path:
        dataset_name = pre_train_model_path
        feature_type, gnn_type, hid_dim, num_layer = 'text', 'GCN', 128, 2
    else:
        dataset_name = pre_train_model_path.split('pre_trained_model/')[1].split('/')[0]
        feature_type, _, gnn_type, hid_dim, num_layer = pre_train_model_path.split(dataset_name+'/')[1].split('.')[:-1]

    print('Pre_train model settings: ')
    print('dataset_name: ', dataset_name)
    print('feature_type: ', feature_type)
    print('gnn_type: ', gnn_type)
    print('hid_dim: ', hid_dim)
    print('num_layer: ', num_layer)

    hid_dim = int(hid_dim)
    num_layer = int(num_layer)
    
    # batch run
    res_file = open("results_1shot.txt", "a")
    for batch_dataset in ["COX2"]: #["COX2", "BZR", "ENZYMES", "PROTEINS", "DD", "MUTAG"]:
        batch_pre_train_model_path = "/home/jz875/project/label_prompt/pre_trained_model/"+batch_dataset+"/text.GraphMAE.GCN.128.2.pth"
        feature_type, _, gnn_type, hid_dim, num_layer = batch_pre_train_model_path.split(batch_dataset+'/')[1].split('.')[:-1]
        data, input_dim, output_dim = load4graph(dataname = batch_dataset, feature_type = feature_type)
        hid_dim = int(hid_dim)
        num_layer = int(num_layer)
        for batch_prompt_type in ["label_prompt"]: #["label_prompt"]: #["GPF", "Gprompt", "GPF-plus", "All-in-one"]:

            tasker = GraphTask(pre_train_model_path = batch_pre_train_model_path, 
                    dataset_name = batch_dataset, num_layer = num_layer,
                    gnn_type = gnn_type, hid_dim = hid_dim, task_train_type = batch_prompt_type,
                    epochs = args.epochs, shot_num = shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = args.batch_size, downstream_task = downstream_task, data = data, input_dim = input_dim, output_dim = output_dim, task_num = args.task_num, feature_type = feature_type)
            tasker.run(res_file)
    res_file.close()

    
    '''
    tasker = GraphTask(pre_train_model_path = pre_train_model_path, 
                    dataset_name = dataset_name, num_layer = num_layer,
                    gnn_type = gnn_type, hid_dim = hid_dim, task_train_type = task_train_type,
                    epochs = args.epochs, shot_num = shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = args.batch_size, downstream_task = downstream_task, data = data, input_dim = input_dim, output_dim = output_dim, task_num = args.task_num, feature_type = feature_type)
    
    pre_train_type = tasker.pre_train_type
    tasker.run()
    '''
