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
from .constant import COLS, CLASS_ACTION_MAP, BUY_IDX, NONE_IDX, SELL_IDX
from .data_utils import get_loader
from .model import GruClassifier 
from .metric import MetricTracker
from .profit_calculator import check_stock_actions_length, calculate_profit


SEED = 0
random.seed(SEED)
np.random.seed(SEED)


class TraderMode(IntEnum):
    TRAINING = 0
    TESTING = 1


class BaseTrader:
    def __init__(self, normalizer_name='quantile'):
        self.mode = TraderMode.TRAINING

    def get_normalizer(self, normalizer_name):
        if normalizer_name == 'quantile':
            return QuantileTransformer(output_distribution='normal')
        elif normalizer_name == 'minmax':
            return MinMaxScaler()

    def train(self, data):
        raise NotImplementedError
    
    def predict_action(self, row):
        raise NotImplementedError


class Trader(BaseTrader):
    def __init__(self, 
                 epochs=10, batch_size=128, 
                 lr=1e-3, model_hidden_size=32,
                 num_past=30, num_pred=20, 
                 normalizer_name='quantile'):
        super().__init__(normalizer_name)

        self.normalizer = self.get_normalizer(normalizer_name)
        self.num_past = num_past
        self.num_pred = num_pred
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_hidden_size = model_hidden_size
        self.lr = lr
        
    def train(self, data):
        self.mode = TraderMode.TRAINING
        
        # get data
        self.raw_data = deepcopy(data)

        # add moving average
        tmp = pd.DataFrame(data=self.raw_data, columns=list(range(4)))
        for window in [3, 5, 7, 10, 15, 20]:
            tmp[f'ma{window}'] = tmp[0].rolling(window).mean()
        self.raw_data = tmp.dropna().values
        del tmp

        # split train, valid
        train_data = deepcopy(self.raw_data[:-self.num_past])
        valid_data = deepcopy(self.raw_data[-(self.num_past+self.num_pred):])

        # normalize
        self.normalizer.fit(train_data)
        train_data = self.normalizer.transform(train_data)
        valid_data = self.normalizer.transform(valid_data)
        
        # get dataloader
        train_loader, valid_loader = get_loader(
            train_data, valid_data,
            self.num_past, self.num_pred, self.batch_size
        )

        # training setup
        self.model = GruClassifier(
            num_feat=train_data.shape[-1],
            hidden_size=self.model_hidden_size
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        criterion = nn.CrossEntropyLoss()

        # core metric
        best_profit = float("-inf")
        val_loss = float("inf")

        # start training
        for epoch in range(self.epochs):

            # print(f'=== epoch: {epoch} ===')

            self.model.train()
            mt = MetricTracker(num_pred=self.num_pred)
            # pbar = tqdm(train_loader, ncols=100)
            for x, class_y, raw_y in train_loader:
                # prediction
                pred, _ = self.model(x)
                
                # the prediction is [-num_pred: -1)], 
                # the last prediction is redundant
                pred = pred[:, -(self.num_pred):-1].reshape(-1, 3)
                class_y = class_y.view(-1)

                loss = criterion(pred, class_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            ## print training metric 
            #     sizes = raw_y.size()
            #     raw_y = raw_y.numpy().reshape(-1, sizes[-1])
            #     raw_y = self.normalizer.inverse_transform(raw_y)
            #     raw_y = raw_y.reshape(*sizes)
            #     mt.update(pred, class_y, raw_y, loss.item())
            # print(f'Train epoch: {epoch}, {mt}')

            self.model.eval()
            mt = MetricTracker(num_pred=self.num_pred)
            for x, class_y, raw_y in valid_loader:
                with torch.no_grad():
                    pred, _ = self.model(x)
                    pred = pred[:, -(self.num_pred):-1].reshape(-1, 3)
                    class_y = class_y.view(-1)
                    loss = criterion(pred, class_y)

                # build raw_y to compute val_profit
                sizes = raw_y.size()
                raw_y = raw_y.numpy().reshape(-1, sizes[-1])
                raw_y = self.normalizer.inverse_transform(raw_y)
                raw_y = raw_y.reshape(*sizes)
                
                mt.update(pred, class_y, raw_y, loss.item())

                if (mt.profit / mt.count) > best_profit:
                    best_profit = (mt.profit / mt.count)
                #     torch.save(self.model.state_dict(), "model.pth")
                
                if loss.item() < val_loss:
                    val_loss = loss.item()
                    torch.save(self.model.state_dict(), "model.pth")

            print(f'Valid epoch: {epoch}, {mt}')
        
        print('=== Stop Training ===')
        print(f'Best val_profit: {best_profit}, val_loss: {val_loss}')
    
    def predict_action(self, row):
        
        # first prediction
        if self.mode == TraderMode.TRAINING:
            self.mode = TraderMode.TESTING

            # load model
            self.model.load_state_dict(torch.load("model.pth"))
            self.model.eval()

            # get current hidden state from previous raw_data
            with torch.no_grad():
                pre_x = self.normalizer.transform(self.raw_data[-(self.num_past):])
                pre_x = torch.tensor(pre_x).unsqueeze(0).float()
                _, ht = self.model(pre_x)
                self.ht = ht.detach().numpy()
            
            # this variable generate action
            self.cur_sum = 0
            print('origin action: ')

        # add moving average to row
        row = list(row)
        for window in [3, 5, 7, 10, 15, 20]:
            # bp()
            row.append(np.mean([row[0]] + list(self.raw_data[-window+1:, 0])))
        row = np.array(row).reshape(1, -1)

        # append new row to raw_data
        self.raw_data = np.concatenate((self.raw_data, row), 0)
        row = self.normalizer.transform(row)
        
        # get previous hidden state
        h0 = torch.tensor(self.ht)
        # bp()
        # predict
        with torch.no_grad():
            row = torch.tensor(row).unsqueeze(0).float()
            pred, ht = self.model(row, h0)

        # set ht to current hidden state for next iteration
        self.ht = ht.detach().numpy()

        # generate action
        pred = pred.detach().numpy().reshape(-1)
        action = CLASS_ACTION_MAP[np.argmax(pred)]
        print(action, end=' ')
        if abs(self.cur_sum + action) >= 2:
            action = 0
        self.cur_sum += action

        return str(action) + '\n'
    
    def re_training(self):
        pass




            
        