import torch
from torch_geometric.loader import DataLoader
from task_strategies import base
import time
import warnings
import numpy as np
from data import load4node_edge, node_sample_and_save, split_induced_graphs, GraphDataset
import pickle
import os
warnings.filterwarnings("ignore")

import torchmetrics
import torch
import random
from tqdm import tqdm
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

class NodeTask(base.BaseTask):
    def __init__(self, data, input_dim, output_dim, task_num = 5, graphs_list = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.task_num = task_num
        self.feature_type = feature_type
        
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
                    node_sample_and_save(self.data, k, folder, self.output_dim)
                    print(str(k) + ' shot ' + str(i) + ' th is saved!!')

    def MLP_finetune_train(self, data, train_idx):
        self.gnn.train()
        self.answering.train()
        self.optimizer.zero_grad() 
        out = self.gnn(data.x, data.edge_index, batch=None) 
        out = self.answering(out)
        loss = self.criterion(out[train_idx], data.y[train_idx])
        loss.backward()  
        self.optimizer.step()  
        return loss.item()
    
    def label_prompt_train(self, data, train_idx):
        self.prompt.train()
        self.optimizer.zero_grad() 
        critical_emb = self.prompt.get_critical_emb(self.gnn, data.x, data.edge_index)
        loss = self.criterion(self.prompt.answering(critical_emb)[train_idx], data.y[train_idx])
        max_loss, min_loss, noise_loss = self.prompt.MI_loss(self.gnn, data, train_idx)
        loss += max_loss + noise_loss + min_loss
        
        loss.backward()  
        self.optimizer.step()
        return loss.item()
    
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

    def GPPTtrain(self, data, train_idx):
        self.prompt.train()
        node_embedding = self.gnn(data.x, data.edge_index)
        out = self.prompt(node_embedding, data.edge_index)
        loss = self.criterion(out[train_idx], data.y[train_idx])
        loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
        self.pg_opi.zero_grad()
        loss.backward()
        self.pg_opi.step()
        mid_h = self.prompt.get_mid_h()
        self.prompt.update_StructureToken_weight(mid_h)
        return loss.item()
    
    def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
        #we update answering and prompt alternately.
        # tune task head
        self.answering.train()
        self.prompt.eval()
        self.gnn.eval()
        for epoch in range(1, answer_epoch + 1):
            answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
            print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

        # tune prompt
        self.answering.eval()
        self.prompt.train()
        for epoch in range(1, prompt_epoch + 1):
            pg_loss = self.prompt.Tune(train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
            print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, prompt_epoch, pg_loss)))
        
        # return pg_loss
        return answer_loss
    
    def GpromptTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0 
        accumulated_centers = None
        accumulated_counts = None
        for batch in train_loader:  
            self.pg_opi.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, task_train_type = 'Gprompt')
            # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
            center, class_counts = center_embedding(out, batch.y, self.output_dim)
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
    
    def run(self, f):
        test_metrics = dict()
        batch_best_loss = []
        if self.task_train_type == 'All-in-one':
            self.answer_epoch = 50
            self.prompt_epoch = 50
            self.epochs = int(self.epochs/self.answer_epoch)
        for i in range(1, 6):
            train_graphs, test_graphs = self.load_induced_graph(i) 
            train_dataset = GraphDataset(train_graphs)
            test_dataset = GraphDataset(test_graphs)

            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            print("prepare induce graph data is finished!")

            self.initialize_gnn()
            self.answering = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim)).to(self.device) 
            self.initialize_prompt()
            self.initialize_optimizer()
            
            idx_train = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i)).type(torch.long).to(self.device)
            print('idx_train',idx_train)
            train_lbls = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
            print("true",i,train_lbls)
            
            idx_test = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i)).type(torch.long).to(self.device)
            test_lbls = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
            
            if self.task_train_type == 'GPPT':
                node_embedding = self.gnn(self.data.x, self.data.edge_index)
                self.prompt.weigth_init(node_embedding,self.data.edge_index, self.data.y, idx_train)

            patience = 20
            best = 1e9
            cnt_wait = 0
            best_loss = 1e9

            for epoch in range(1, self.epochs):
                t0 = time.time()
                center = None
                if self.task_train_type == 'MLP_finetune':
                    loss = self.MLP_finetune_train(self.data, idx_train) 
                elif self.task_train_type == 'label_prompt':  
                    loss = self.label_prompt_train(self.data, idx_train)
                elif self.task_train_type == 'GPPT':
                    loss = self.GPPTtrain(self.data, idx_train)                
                elif self.task_train_type == 'All-in-one':
                    loss = self.AllInOneTrain(train_loader,self.answer_epoch,self.prompt_epoch)                           
                elif self.task_train_type in ['GPF', 'GPF-plus']:
                    loss = self.GPFTrain(train_loader)                                                          
                elif self.task_train_type =='Gprompt':
                    loss, center = self.GpromptTrain(train_loader)

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
                    metric_dict = self.prompt.Eva_Node(test_loader, self.data, idx_test, idx_train, self.gnn, self.answering,self.output_dim, self.device, center)
                else:
                    metric_dict = self.prompt.Eva_Node(test_loader, self.data, idx_test, self.gnn, self.answering,self.output_dim, self.device, center)
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

    def load_induced_graph(self, i_th):

        idx_train = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i_th)).type(torch.long).to(self.device)
        if self.downstream_task in ['avg_socre', 'cite_num']:
            train_lbls = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i_th)).type(torch.float).squeeze().to(self.device)
        else:
            train_lbls = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i_th)).type(torch.long).squeeze().to(self.device)
        
        idx_test = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i_th)).type(torch.long).to(self.device)
        if self.downstream_task in ['avg_socre', 'cite_num']:
            test_lbls = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i_th)).type(torch.float).squeeze().to(self.device)
        else:
            test_lbls = torch.load("datasets/few_shot_data/{}/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.downstream_task, self.shot_num, i_th)).type(torch.long).squeeze().to(self.device)


        folder_path = "datasets/induced_graph/{}/{}/{}_shot/{}/".format(self.dataset_name, self.downstream_task, self.shot_num, i_th)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print([len(idx_train), len(idx_test)])
        file_path = folder_path + '/induced_graph_min10_max30'+'_'+str(self.feature_type)+'.pkl'
        if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    print('loading induced graph...')
                    train_graphs, test_graphs = pickle.load(f)
                    print('Done!!!')
        else:
            print('Begin split_induced_graphs.')
            split_induced_graphs(self.data, folder_path, self.device, idx_train, idx_test, smallest_size=10, largest_size=30, task_type = 'node', feature_type = self.feature_type)
            with open(file_path, 'rb') as f:
                train_graphs, test_graphs = pickle.load(f)
        
        train_graphs = [graph.to(self.device) for graph in train_graphs]
        test_graphs = [graph.to(self.device) for graph in test_graphs]
        print([len(train_graphs), len(test_graphs)])
        return train_graphs, test_graphs

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
    res_file = open("results_1shot_node.txt", "a")
    for batch_dataset in ["chameleon"]: #["Cora", "PubMed", "CiteSeer", "chameleon", "ENZYMES", "Actor", "squirrel", "Computers"]: #["Cora", "PubMed", "CiteSeer", "chameleon", "ENZYMES", "Actor"]:
        batch_pre_train_model_path = "/home/jz875/project/label_prompt/pre_trained_model/"+batch_dataset+"/text.GraphMAE.GCN.128.2.pth"
        if '.pth' not in batch_pre_train_model_path:
            dataset_name = batch_dataset
            feature_type, gnn_type, hid_dim, num_layer = 'text', 'GCN', 128, 2
        else:
            feature_type, _, gnn_type, hid_dim, num_layer = batch_pre_train_model_path.split(batch_dataset+'/')[1].split('.')[:-1]
        data, input_dim, output_dim = load4node_edge(dataname = batch_dataset, feature_type = feature_type, task_name = downstream_task)

        data = data.to(args.device)
        hid_dim = int(hid_dim)
        num_layer = int(num_layer)
        for batch_prompt_type in ["label_prompt"]: #["GPF", "Gprompt", "GPF-plus", "All-in-one"]:

            tasker = NodeTask(pre_train_model_path = batch_pre_train_model_path, 
                    dataset_name = batch_dataset, num_layer = num_layer,
                    gnn_type = gnn_type, hid_dim = hid_dim, task_train_type = batch_prompt_type,
                    epochs = args.epochs, shot_num = shot_num, device = args.device, lr = args.lr, wd = args.decay,
                    batch_size = args.batch_size, downstream_task = downstream_task, data = data, input_dim = input_dim, output_dim = output_dim, task_num = args.task_num, feature_type = feature_type)
            tasker.run(res_file)
    res_file.close()
    
    '''
    tasker = NodeTask(pre_train_model_path = pre_train_model_path, 
                    dataset_name = dataset_name, num_layer = num_layer,
                    gnn_type = gnn_type, hid_dim = hid_dim, task_train_type = task_train_type,
                    epochs = args.epochs, shot_num = shot_num, device=args.device, lr = args.lr, wd = args.decay,
                    batch_size = args.batch_size, downstream_task = downstream_task, data = data, input_dim = input_dim, output_dim = output_dim, task_num = args.task_num, feature_type = feature_type)
    
    pre_train_type = tasker.pre_train_type
    tasker.run()
    '''
    

