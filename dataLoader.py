import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class TradeDataset(Dataset):
    def __init__(self, filePath):
        xy = np.loadtxt(filePath, delimiter=',', dtype=np.float32)
        self.n_samples = xy.shape[0]

        self.x_data = torch.from_numpy(xy[:, :2400])
        self.y_data = torch.from_numpy(xy[:, [2400]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
