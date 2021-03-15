import torch
import os


class ContrastiveTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    def train_best_model(self, epochs, train_loader, val_loader,
                         checkpoint_folder, fold=1, path_best_model='', save_model=True):
        acc_dict = {'train': [], 'val': []}
        best_acc = 0
        best_epoch = 0
        for epoch in epochs:
            self.train_epoch(train_loader)
            trn_acc = self.evaluate_epoch(train_loader)
            val_acc = self.evaluate_epoch(val_loader)
            acc_dict['train'].append(trn_acc)
            acc_dict['val'].append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                if save_model:
                    self.save_checkpoint(best_epoch, fold, acc_dict, checkpoint_folder,
                                         path_best_model)

        return acc_dict, best_epoch

    def train_epoch(self, dataloader):
        self.model.train()
        for (x_p, x_q, y_i), _ in dataloader:
            x_p, x_q = x_p.to(self.device), x_q.to(self.device)
            y_i = y_i.to(self.device)
            self.optimizer.zero_grad()
            phi_x_p = self.model(x_p)  # embeddings of x_p
            # z_x_p = torch.sqrt(torch.sum(phi_x_p ** 2, dim=1, keepdim=True))   normalizing
            # phi_x_p /= z_x_p
            phi_x_q = self.model(x_q)  # embeddings of x_q
            # z_x_q = torch.sqrt(torch.sum(phi_x_q ** 2, dim=1, keepdim=True))   normalizing
            # phi_x_q /= z_x_q
            loss = self.criterion(phi_x_p, phi_x_q, y_i)
            loss.backward()
            self.optimizer.step()

    def evaluate_epoch(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            total_acc = 0.0
            data_size = 0
            for (x_p, _, _), targets in dataloader:
                x_p, targets = x_p.to(self.device), targets.to(self.device)
                output = self.model(x_p, predict_class=True)  # log probs.
                preds = torch.argmax(output, dim=1, keepdim=True)  # (B, 1) instead of (B,)
                data_size += x_p.size(0)
                acc = torch.eq(preds, targets.view_as(preds)).sum().item()
                total_acc += acc
        self.model.train()
        total_acc /= data_size
        return total_acc

    def save_checkpoint(self, epoch, fold, acc_dict, checkpoint_folder, path_model):
        model_saved_name = f"{path_model}_fold={fold}_epoch={epoch}"
        full_path = os.path.join(checkpoint_folder,
                                 f'{model_saved_name}.pth.tar')
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'acc_dict': acc_dict}, full_path)

    def load_model(self, path_model):
        checkpoint = torch.load(path_model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        dict_acc = checkpoint['acc_dict']
        return dict_acc


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    def train_best_model(self, epochs, train_loader, val_loader,
                         checkpoint_folder, fold=1, path_best_model='', save_model=True):
        acc_dict = {'train': [], 'val': []}
        best_acc = 0
        best_epoch = 0
        for epoch in epochs:
            self.train_epoch(train_loader)
            trn_acc = self.evaluate_epoch(train_loader)
            val_acc = self.evaluate_epoch(val_loader)
            acc_dict['train'].append(trn_acc)
            acc_dict['val'].append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                if save_model:
                    self.save_checkpoint(best_epoch, fold, acc_dict, checkpoint_folder,
                                         path_best_model)

        return acc_dict, best_epoch

    def train_epoch(self, dataloader):
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)  #
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

    def evaluate_epoch(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            total_acc = 0.0
            data_size = 0
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)  # log probs.
                preds = torch.argmax(output, dim=1, keepdim=True)  # (B, 1) instead of (B,)
                data_size += X.size(0)
                # print(f'y = {y}')
                # print(f'preds = {preds}')
                acc = torch.eq(preds, y.view_as(preds)).sum().item()
                # print(f'acc = {acc}')
                # print(f'batch_size  = {X.size(0)}')
                total_acc += acc
        self.model.train()
        total_acc /= data_size
        return total_acc

    def save_checkpoint(self, epoch, fold, acc_dict, checkpoint_folder, path_model):
        model_saved_name = f"{path_model}_fold={fold}_epoch={epoch}"
        full_path = os.path.join(checkpoint_folder,
                                 f'{model_saved_name}.pth.tar')
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'acc_dict': acc_dict}, full_path)

    def load_model(self, path_model):
        checkpoint = torch.load(path_model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        dict_acc = checkpoint['acc_dict']
        return dict_acc
