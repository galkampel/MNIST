import torch
import os


class Trainer:
    def __init__(self, model, optimizer, criterion, device, criterion_type='standard', criterion_cls=None,
                 normalize=False):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.normalize = normalize
        self.criterion_cls = criterion_cls
        trn_func_dict = {'standard': self.train_epoch_standard, 'contrastive': self.train_epoch_contrastive,
                          'triplet': self.train_epoch_triplet}
        self.train_epoch = trn_func_dict[criterion_type]
        eval_func_dict = {'standard': self.eval_epoch_standard, 'contrastive': self.eval_epoch_contrastive,
                         'triplet': self.eval_epoch_triplet}
        self.evaluate_epoch = eval_func_dict[criterion_type]

    def train_epoch_contrastive(self, dataloader):
        self.model.train()
        for (x_p, x_q, y_i), targets in dataloader:
            x_p, x_q = x_p.to(self.device), x_q.to(self.device)
            y_i = y_i.to(self.device)
            self.optimizer.zero_grad()
            phi_x_p = self.model(x_p)  # embeddings of x_p
            phi_x_q = self.model(x_q)  # embeddings of x_q
            loss = self.criterion(phi_x_p, phi_x_q, y_i, self.normalize)
            if self.criterion_cls is not None:
                logits = self.model.fc(phi_x_p)
                loss_cls = self.criterion_cls(logits, targets.to(self.device))  # CrossEntropyLoss
                loss += loss_cls
            loss.backward()
            self.optimizer.step()

    def train_epoch_triplet(self, dataloader):
        self.model.train()
        for (x_a, x_p, x_n), targets in dataloader:
            x_a, x_q = x_a.to(self.device)
            x_p, x_n = x_p.to(self.device), x_n.to(self.device)
            self.optimizer.zero_grad()
            phi_x_a = self.model(x_a)  # embeddings of x_a
            phi_x_p = self.model(x_p)  # embeddings of x_p
            phi_x_n = self.model(x_n)  # embeddings of x_n
            loss = self.criterion(phi_x_a, phi_x_p, phi_x_n, self.normalize)
            if self.criterion_cls is not None:
                logits = self.model.fc(phi_x_p)
                loss_cls = self.criterion_cls(logits, targets.to(self.device))  # CrossEntropyLoss
                loss += loss_cls
            loss.backward()
            self.optimizer.step()

    def train_epoch_standard(self, dataloader):
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X, predict_class=True)  #
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

    def eval_epoch_contrastive(self, dataloader):
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

    def eval_epoch_triplet(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            total_acc = 0.0
            data_size = 0
            for (x_a, _, _), targets in dataloader:
                x_a, targets = x_a.to(self.device), targets.to(self.device)
                output = self.model(x_a, predict_class=True)  # log probs.
                preds = torch.argmax(output, dim=1, keepdim=True)  # (B, 1) instead of (B,)
                data_size += x_a.size(0)
                acc = torch.eq(preds, targets.view_as(preds)).sum().item()
                total_acc += acc
        self.model.train()
        total_acc /= data_size
        return total_acc

    def eval_epoch_standard(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            total_acc = 0.0
            data_size = 0
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X, predict_class=True)  # log probs.
                preds = torch.argmax(output, dim=1, keepdim=True)  # (B, 1) instead of (B,)
                data_size += X.size(0)
                acc = torch.eq(preds, y.view_as(preds)).sum().item()

                total_acc += acc
        self.model.train()
        total_acc /= data_size
        return total_acc

    def save_checkpoint(self, epoch, fold, acc_dict, model_name, checkpoint_folder='checkpoint'):
        model_saved_name = f"{model_name}_fold={fold}"
        folder_path = os.path.join(os.getcwd(), checkpoint_folder)
        os.makedirs(folder_path, exist_ok=True)
        full_path = os.path.join(folder_path,
                                 f'{model_saved_name}.pt')
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'acc_dict': acc_dict,
                    'epoch': epoch}, full_path)

    def load_model(self, model_name, model_dir_name='checkpoint'):
        model_dir = os.path.join(os.getcwd(), model_dir_name)

        model_path = os.path.join(model_dir, f'{model_name}.pt')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        dict_acc = checkpoint['acc_dict']
        epoch = checkpoint['epoch']
        return dict_acc, epoch
