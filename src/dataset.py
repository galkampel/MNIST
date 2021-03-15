import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset


def load_MNIST_dataset(transform, train, root='~/MNIST/', download=True):
    return datasets.MNIST(
        root=root, train=train, download=download, transform=transform
    )


class MNISTContrastiveDataset(Dataset):
    def __init__(self, dataset, dataset_idxes, distribution_dict, mean=33.31, sd=78.5675, seed=1):
        self.X = dataset.data.numpy()[dataset_idxes]
        self.X = (self.X - mean) / sd  # normalizing X
        self.y = dataset.targets.numpy()[dataset_idxes]
        self.distribution_dict = distribution_dict  # key=label, vale=dist. of similarities to label
        self.index_arr = np.arange(self.y.size)
        self.y_vals = np.unique(self.y)
        np.random.seed(seed)

    def __len__(self):
        return self.y.size

    def __getitem__(self, index):
        x_p = torch.tensor(self.X[index], dtype=torch.float).unsqueeze(0)
        label = self.y[index]
        x_q, y_i = self.set_x_q(label, index)
        x_q = torch.tensor(x_q, dtype=torch.float).unsqueeze(0)
        y_i = torch.tensor(y_i, dtype=torch.float)  # dtype=torch.uint8
        # y_i = torch.tensor(y_i, dtype=torch.uint8)
        label = torch.tensor(label, dtype=torch.long)
        return (x_p, x_q, y_i), label

    def set_x_q(self, label, index):
        label_dist = self.distribution_dict[label]
        cand_label = np.random.choice(self.y_vals, p=label_dist)
        y_i = int(label == cand_label)
        rel_idxes = self.index_arr[self.y == cand_label]
        return self.X[np.random.choice(rel_idxes)], y_i


class MNISTTripletDataset(Dataset):
    def __init__(self, dataset, dataset_idxes, distribution_dict, mean=33.31, sd=78.5675, seed=1):
        self.X = dataset.data.numpy()[dataset_idxes]
        self.X = (self.X - mean) / sd  # normalizing X
        self.y = dataset.targets.numpy()[dataset_idxes]
        self.distribution_dict = distribution_dict  # key=label, vale=dist. of similarities to label
        self.index_arr = np.arange(self.y.size)
        self.y_vals = np.unique(self.y)
        np.random.seed(seed)

    def __len__(self):
        return self.y.size

    def __getitem__(self, index):
        anchor = torch.tensor(self.X[index], dtype=torch.float).unsqueeze(0)
        label = self.y[index]
        pos_example = self.set_positive_example(label, index)
        pos_example = torch.tensor(pos_example, dtype=torch.float).unsqueeze(0)
        neg_example = self.set_negative_example(label)
        neg_example = torch.tensor(neg_example, dtype=torch.float).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.uint8)
        return (anchor, pos_example, neg_example), label

    def set_positive_example(self, label, index):
        rel_idxes = self.index_arr[(self.y == label) & (self.index_arr != index)]
        return self.X[np.random.choice(rel_idxes)]

    def set_negative_example(self, label):
        neg_label_dist = self.distribution_dict[label]
        neg_label = np.random.choice(self.y_vals, p=neg_label_dist)
        rel_idxes = self.index_arr[self.y == neg_label]
        return self.X[np.random.choice(rel_idxes)]
