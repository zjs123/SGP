import torch
import torch.nn.functional as F
import torch.nn as nn

import torchmetrics

class Gprompt_tuning_loss(nn.Module):
    def __init__(self, tau=0.1):
        super(Gprompt_tuning_loss, self).__init__()
        self.tau = tau
    
    def forward(self, embedding, center_embedding, labels):
        # 对于每个样本对（xi,yi), loss为 -ln(sim正 / sim正+sim负)

        # 计算所有样本与所有个类原型的相似度
        similarity_matrix = F.cosine_similarity(embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1) / self.tau
        exp_similarities = torch.exp(similarity_matrix)
        # Sum exponentiated similarities for the denominator
        pos_neg = torch.sum(exp_similarities, dim=1, keepdim=True)
        # select the exponentiated similarities for the correct classes for the every pair (xi,yi)
        pos = exp_similarities.gather(1, labels.view(-1, 1))
        L_prompt = -torch.log(pos / pos_neg)
        loss = torch.sum(L_prompt)
                    
        return loss

class Gprompt(torch.nn.Module):
    def __init__(self,input_dim):
        super(Gprompt, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, node_embeddings):
        node_embeddings=node_embeddings*self.weight
        return node_embeddings
    
    def Eva_Node(self, loader, data, idx_test,  gnn, answering, num_class, device, center = None):
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
                out = gnn(batch.x, batch.edge_index, batch.batch, self.forward, 'Gprompt')
                similarity_matrix = F.cosine_similarity(out.unsqueeze(1), center.unsqueeze(0), dim=-1)
                pred = similarity_matrix.argmax(dim=1)
                acc = accuracy(pred, batch.y)
                ma_f1 = macro_f1(pred, batch.y)
                mi_f1 = micro_f1(pred, batch.y)
                wei_f1 = weighted_f1(pred, batch.y)
                roc = auroc(similarity_matrix, batch.y)
                prc = auprc(similarity_matrix, batch.y)

        acc = accuracy.compute()
        macro = macro_f1.compute()
        micro = micro_f1.compute()
        weighted = weighted_f1.compute()
        roc = auroc.compute()
        prc = auprc.compute()  
        return {'task_prompt': 'Gprompt', 'ACC':acc.item(), 'Macro_F1':macro.item(), 'Micro_F1':micro.item(), 'Weighted_F1':weighted.item(), 'ROC':roc.item(), 'PRC':prc.item()}
        
    def Eva_Graph(self, loader, data, idx_test,  gnn, answering, num_class, device, center = None):
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
                out = gnn(batch.x, batch.edge_index, batch.batch, self.forward, 'Gprompt')
                similarity_matrix = F.cosine_similarity(out.unsqueeze(1), center.unsqueeze(0), dim=-1)
                pred = similarity_matrix.argmax(dim=1)
                acc = accuracy(pred, batch.y)
                ma_f1 = macro_f1(pred, batch.y)
                mi_f1 = micro_f1(pred, batch.y)
                wei_f1 = weighted_f1(pred, batch.y)
                roc = auroc(similarity_matrix, batch.y)
                prc = auprc(similarity_matrix, batch.y)

        acc = accuracy.compute()
        macro = macro_f1.compute()
        micro = micro_f1.compute()
        weighted = weighted_f1.compute()
        roc = auroc.compute()
        prc = auprc.compute()  
        return {'task_prompt': 'Gprompt', 'ACC':acc.item(), 'Macro_F1':macro.item(), 'Micro_F1':micro.item(), 'Weighted_F1':weighted.item(), 'ROC':roc.item(), 'PRC':prc.item()}