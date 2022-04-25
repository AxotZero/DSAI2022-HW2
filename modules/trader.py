# std lib
from enum import IntEnum
import random
from copy import deepcopy
from pdb import set_trace as bp

# 3rd lib
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
import pandas as pd
 
# my lib
from .constant import COLS, BUY_IDX, NONE_IDX, SELL_IDX
from .model import GruClassifier, GruRatioRegressor
from .profit_calculator import check_stock_actions_length, calculate_profit


SEED = 0
random.seed(SEED)
np.random.seed(SEED)


class TraderMode(IntEnum):
    TRAINING = 0
    TESTING = 1


class Trader:
    def __init__(self, 
                 epochs=400, lr=1e-2, model_hidden_size=16,
                 num_past=20):

        self.num_past = num_past
        self.epochs = epochs
        self.model_hidden_size = model_hidden_size
        self.lr = lr
        
    def train(self, data):
        self.mode = TraderMode.TRAINING
        
        # get data
        # self.x_normalizer = MinMaxScaler(feature_range=(-1, 1))
        self.x_normalizer = QuantileTransformer(output_distribution='normal')
        self.raw_data = deepcopy(data)
        self.raw_data = self.x_normalizer.fit_transform(self.raw_data)
        
        # generate x, y
        datas = []
        targets = []
        for i in range(len(self.raw_data) - self.num_past - 1):
            datas.append(self.raw_data[i: i+self.num_past])
            target1 = self.raw_data[i+1: i+self.num_past+1, [0]]
            target2 = self.raw_data[i+2: i+self.num_past+2, [0]]
            target = np.concatenate([target1, target2], axis=1)
            targets.append(target)
            # bp()

        datas = np.array(datas)
        targets = np.array(targets)

        # train_x = torch.tensor(datas).float()
        # train_y = torch.tensor(targets).float()

        # shuffle
        idx_full = np.arange(len(datas))
        np.random.seed(0)
        np.random.shuffle(idx_full)
        datas = datas[idx_full]
        targets = targets[idx_full]

        # train_valid split
        valid_ratio = 0.2
        valid_size = round(valid_ratio * datas.shape[0])
        train_size = datas.shape[0] - valid_size

        train_x, train_y = datas[:train_size], targets[:train_size]
        valid_x, valid_y = datas[train_size:], targets[train_size:]


        # convert to tensor
        train_x = torch.tensor(train_x).float()
        train_y = torch.tensor(train_y).float()
        valid_x = torch.tensor(valid_x).float()
        valid_y = torch.tensor(valid_y).float()

        print(train_x.size())
        print(train_y.size())
        print(valid_x.size())
        print(valid_y.size())

        # training setup
        self.model = GruRatioRegressor(
            num_feat=4,
            num_out= 2,
            hidden_size=self.model_hidden_size,
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        criterion = nn.MSELoss()

        # core metric
        best_val_loss = float("inf")
        best_epoch = -1

        # start training
        pbar = tqdm(range(self.epochs), ncols=80)
        for epoch in pbar:

            # train
            self.model.train()
            pred, _ = self.model(train_x)
            # bp()
            loss = criterion(pred, train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(f'Train epoch: {epoch}, mse: {loss.item() :.6f}')

            # valid
            self.model.eval()
            pred, _ = self.model(valid_x)
            loss = criterion(pred, valid_y)
            # print(f'Valid epoch: {epoch}, mse: {loss.item() :.6f}')

            pbar.set_description(f'epoch: {epoch}, mse: {loss.item():.6f}')
            pbar.update()
            
            # check whether get better result
            if loss.item() < best_val_loss:
                best_val_loss = loss.item()
                best_epoch = epoch
                torch.save(self.model.state_dict(), "./data/model_data/model.pth")

    
        print('=== Stop Training ===')
        print(f'Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}')
    
    def predict_action(self, row):
        
        # first prediction
        if self.mode == TraderMode.TRAINING:
            self.mode = TraderMode.TESTING

            # load model
            self.model.load_state_dict(torch.load("./data/model_data/model.pth"))
            self.model.eval()

            self.cur_sum = 0


        # append new row to raw_data
        row = np.array(row).reshape(1, -1)
        row = self.x_normalizer.transform(row)
        self.raw_data = np.concatenate((self.raw_data, row), 0)
        _input = self.raw_data[-self.num_past:]
        
        # predict
        with torch.no_grad():
            _input = torch.tensor(_input).unsqueeze(0).float()
            pred, ht = self.model(_input, )


        # generate action
        pred = pred[:, -1].detach().squeeze().numpy()
        # action = BUY_IDX if pred[1] > pred[0] else SELL_IDX
        action = BUY_IDX if pred[1] > pred[0] else SELL_IDX
        print(action, end=' ')
        # print(pred)
        if abs(self.cur_sum + action) >= 2:
            action = 0
        self.cur_sum += action

        return str(action) + '\n'
    
    def re_training(self):
        pass





            
        