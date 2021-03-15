import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_similarity_dist_dict(criterion_type='contrastive'):
    similarity_dist_dict = {
        0: [0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.05, 0.3, 0.2],  # 8 , 9, 6
        1: [0.05, 0.0, 0.1, 0.05, 0.2, 0.05, 0.05, 0.3, 0.05, 0.15],  # 7, 4, 9, 2
        2: [0.075, 0.2, 0.0, 0.075, 0.075, 0.075, 0.075, 0.075, 0.2, 0.15],  # 1, 9, (8?)
        3: [0.05, 0.05, 0.05, 0.0, 0.05, 0.25, 0.05, 0.05, 0.4, 0.05],  # 5, 8
        4: [0.05, 0.25, 0.05, 0.05, 0.0, 0.05, 0.05, 0.15, 0.05, 0.3],  # 9, 1, 7?
        5: [0.05, 0.05, 0.05, 0.15, 0.05, 0.0, 0.35, 0.05, 0.2, 0.05],  # 6, 8, 3
        6: [0.2, 0.075, 0.075, 0.075, 0.075, 0.2, 0.0, 0.075, 0.15, 0.075],  # 0, 8, 5
        7: [0.075, 0.3, 0.075, 0.075, 0.1, 0.075, 0.075, 0.0, 0.075, 0.15],  # 1, 4, 9
        8: [0.25, 0.075, 0.075, 0.2, 0.075, 0.1, 0.075, 0.075, 0.0, 0.075],  # 0, 3, 5, (6, 9)?
        9: [0.1, 0.15, 0.1, 0.05, 0.25, 0.05, 0.05, 0.15, 0.1, 0.0]  # 0, 1, 7, 4, 2, 8
    }

    if criterion_type == 'triplet':
        triplet_dist_dict = {label: np.array(dist) for label, dist in similarity_dist_dict.items()}
        return triplet_dist_dict

    elif criterion_type == 'contrastive':
        contrastive_dist_dict = {}
        for label, dist in similarity_dist_dict.items():
            contrastive_dist_dict[label] = np.array(dist) / 2
            contrastive_dist_dict[label][label] = 0.5
        return contrastive_dist_dict


def l2_norm(x, y):
    return ((x - y) ** 2).sum(1)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, metric=l2_norm, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum

    @staticmethod
    def normalize_x(x):
        z = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / z

    def forward(self, x_p, x_q, y, normalize=True):
        if normalize:
            x_p = self.normalize_x(x_p)
            x_q = self.normalize_x(x_q)
        distance = self.metric(x_p, x_q)
        # print(f'min distance = {distance.min()}\tmax distance = {distance.max()}')
        # print(f'mean distance = {distance.mean()}')
        loss = y * distance + (1 - y) * F.relu(self.margin - distance)  # (B)
        loss = self.reduction(loss)  # 1
        # print(f'loss = {loss}')
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, metric=l2_norm, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum

    @staticmethod
    def normalize_x(x):
        z = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / z

    def forward(self, x_a, x_p, x_n, normalize=True):
        if normalize:
            x_a = self.normalize_x(x_a)
            x_p = self.normalize_x(x_p)
            x_n = self.normalize_x(x_n)
        pos_distance = self.metric(x_a, x_p)
        neg_distamce = self.metric(x_a, x_n)
        loss = F.relu(pos_distance - neg_distamce + self.margin)  # (B)
        loss = self.reduction(loss)  # 1
        return loss


def plot_accuracy(acc_dict, folder_plot, model_name, y_label='Accuracy', x_label='Epochs', save_model=True):
    num_epochs = len(acc_dict['train'])
    df_acc = pd.DataFrame(acc_dict)
    df_acc.set_index(pd.Index(range(1, num_epochs + 1)), inplace=True)
    sns.lineplot(data=df_acc)
    title = model_name.split('_fold')[0].replace('model_', 'model\n').replace('_lr', '\nlr')
    plt.title(title)
    plt.ylim()
    plt.ylabel(y_label)
    plt.xlim(1, num_epochs)
    plt.xlabel(x_label)
    # plt.legend()
    plt.tight_layout()
    if save_model:
        plt.savefig(os.path.join(folder_plot, f"{model_name}.png"))
        plt.close()
    else:
        plt.show()
