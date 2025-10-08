import torch
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
import torchmetrics


class GPF(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(GPF, self).__init__()
        self.global_emb = torch.nn.Parameter(torch.Tensor(1,in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        return x + self.global_emb
    
    def Eva_Node(self, loader, data, idx_test,  gnn, answering, num_class, device, center = None):
        gnn.eval()
        if num_class >= 2:
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
                    batch.x = self.add(batch.x)
                    out = gnn(batch.x, batch.edge_index, batch.batch)
                    if answering:
                        out = answering(out)  
                    pred = out.argmax(dim=1)  

                    acc = accuracy(pred, batch.y)
                    ma_f1 = macro_f1(pred, batch.y)
                    mi_f1 = micro_f1(pred, batch.y)
                    wei_f1 = weighted_f1(pred, batch.y)
                    roc = auroc(out, batch.y)
                    prc = auprc(out, batch.y)

                    # print(acc)
            acc = accuracy.compute()
            macro = macro_f1.compute()
            micro = micro_f1.compute()
            weighted = weighted_f1.compute()
            roc = auroc.compute()
            prc = auprc.compute() 
            return {'task_prompt': 'GPF', 'ACC':acc.item(), 'Macro_F1':macro.item(), 'Micro_F1':micro.item(), 'Weighted_F1':weighted.item(), 'ROC':roc.item(), 'PRC':prc.item()}
        
    def Eva_Graph(self, loader, data, idx_test,  gnn, answering, num_class, device, center = None):
        if answering:
            answering.eval()
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
                batch.x = self.add(batch.x)
                out = gnn(batch.x, batch.edge_index, batch.batch)
                if answering:
                    out = answering(out)  
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
        return {'task_prompt': 'GPF', 'ACC':acc.item(), 'Macro_F1':macro.item(), 'Micro_F1':micro.item(), 'Weighted_F1':weighted.item(), 'ROC':roc.item(), 'PRC':prc.item()}

class GPF_plus(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPF_plus, self).__init__()
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p
    
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
                batch.x = self.add(batch.x)
                out = gnn(batch.x, batch.edge_index, batch.batch)
                if answering:
                    out = answering(out)  
                pred = out.argmax(dim=1)  

                acc = accuracy(pred, batch.y)
                ma_f1 = macro_f1(pred, batch.y)
                mi_f1 = micro_f1(pred, batch.y)
                wei_f1 = weighted_f1(pred, batch.y)
                roc = auroc(out, batch.y)
                prc = auprc(out, batch.y)

                # print(acc)
        acc = accuracy.compute()
        macro = macro_f1.compute()
        micro = micro_f1.compute()
        weighted = weighted_f1.compute()
        roc = auroc.compute()
        prc = auprc.compute()
        return {'task_prompt': 'GPF_plus', 'ACC':acc.item(), 'Macro_F1':macro.item(), 'Micro_F1':micro.item(), 'Weighted_F1':weighted.item(), 'ROC':roc.item(), 'PRC':prc.item()}
    def Eva_Graph(self, loader, data, idx_test,  gnn, answering, num_class, device, center = None):
        if answering:
            answering.eval()
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
                batch.x = self.add(batch.x)
                out = gnn(batch.x, batch.edge_index, batch.batch)
                if answering:
                    out = answering(out)  
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
        return {'task_prompt': 'GPF_plus', 'ACC':acc.item(), 'Macro_F1':macro.item(), 'Micro_F1':micro.item(), 'Weighted_F1':weighted.item(), 'ROC':roc.item(), 'PRC':prc.item()}