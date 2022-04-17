from copy import deepcopy
from pdb import set_trace as bp

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .constant import NONE_CLASS_IDX, BUY_CLASS_IDX, SELL_CLASS_IDX


def load_data(path):
    return np.genfromtxt(path, delimiter=',')


class MyDataset(Dataset):
    def __init__(self, data, num_past=30, num_pred=20, training=True):
        self.data = deepcopy(data)
        self.num_past = num_past
        self.num_pred = num_pred

    def __len__(self):
        return len(self.data) - self.num_past - self.num_pred + 1
    
    def __getitem__(self, x_beg_idx):
        x_end_idx = x_beg_idx + self.num_past + self.num_pred
        y_beg_idx = x_beg_idx + self.num_past
        y_end_idx = x_end_idx

        x = self.data[x_beg_idx: x_end_idx]
        
        raw_y = self.data[y_beg_idx: y_end_idx]
        
        # bp()
        class_y = np.zeros(len(raw_y)-1)
        class_y[:-1] = raw_y[2:, 0] - raw_y[1:-1, 0]
        class_y[-1] = raw_y[-1, -1] - raw_y[-1, 0]
        class_y[class_y == 0] = NONE_CLASS_IDX
        class_y[class_y > 0] = BUY_CLASS_IDX
        class_y[class_y < 0] = SELL_CLASS_IDX

        raw_y = raw_y[-self.num_pred:]
        return (
            torch.tensor(x).float(), 
            torch.tensor(class_y).long(), 
            torch.tensor(raw_y).float()
        )


def get_loader(train_data, valid_data, num_past=30, num_pred=20, batch_size=32):

    train_dataset = MyDataset(
        train_data, 
        num_past=num_past, 
        num_pred=num_pred, 
        training=True
    )
    valid_dataset = MyDataset(
        valid_data, 
        num_past=num_past, 
        num_pred=num_pred, 
        training=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False
    )
    return train_loader, valid_loader