# std lib
from enum import IntEnum
import random
from copy import deepcopy

# 3rd lib
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
 
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
                 epochs=100, batch_size=128, num_past=30, num_pred=20, 
                 normalizer_name='quantile'):
        super().__init__(normalizer_name)

        self.normalizer = self.get_normalizer(normalizer_name)
        self.num_past = num_past
        self.num_pred = num_pred
        self.epochs = epochs
        self.batch_size = batch_size
        

    def train(self, data):
        self.mode = TraderMode.TRAINING
        
        # get data
        self.raw_data = deepcopy(data)
        train_data = data[:-self.num_past]
        valid_data = data[-(self.num_past+self.num_pred):]

        # normalize
        self.normalizer.fit(train_data)
        train_data = self.normalizer.transform(train_data)
        valid_data = self.normalizer.transform(valid_data)
        
        # get dataloader
        train_loader, valid_loader = get_loader(
            train_data, valid_data,
            self.num_past, self.num_pred, self.batch_size
        )

        # training
        self.model = GruClassifier(
            num_feat=train_data.shape[-1],
            hidden_size=32
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-2
        )
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):

            print(f'=== epoch: {epoch} ===')

            self.model.train()
            mt = MetricTracker(num_pred=self.num_pred)
            # pbar = tqdm(train_loader, ncols=100)
            for x, class_y, raw_y in train_loader:
                
                pred, _ = self.model(x)

                pred = pred[:, -(self.num_pred-1):].reshape(-1, 3)
                class_y = class_y.view(-1)

                loss = criterion(pred, class_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

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
                    pred = pred[:, -(self.num_pred-1):].reshape(-1, 3)
                    class_y = class_y.view(-1)
                    loss = criterion(pred, class_y)

                sizes = raw_y.size()
                raw_y = raw_y.numpy().reshape(-1, sizes[-1])
                raw_y = self.normalizer.inverse_transform(raw_y)
                raw_y = raw_y.reshape(*sizes)
                
                mt.update(pred, class_y, raw_y, loss.item())

            print(f'Valid epoch: {epoch}, {mt}')
    
    def predict_action(self, row):
        
        # first prediction
        if self.mode == TraderMode.TRAINING:
            self.mode = TraderMode.TESTING
            self.cur_sum = 0
            self.model.eval()

            # get ht
            with torch.no_grad():
                pre_x = self.normalizer.transform(self.raw_data)
                pre_x = torch.tensor(pre_x).unsqueeze(0).float()
                _, ht = self.model(pre_x)
                self.ht = ht.detach().numpy()

        # predict action
        row = np.array(row).reshape(1, -1)
        row = self.normalizer.transform(row)
        row = torch.tensor(row).unsqueeze(0).float()

        h0 = torch.tensor(self.ht)
        with torch.no_grad():
            pred, ht = self.model(row, h0)

        self.ht = ht.detach().numpy()
        pred = pred.detach().numpy().reshape(-1)
        action = CLASS_ACTION_MAP[np.argmax(pred)]

        if abs(self.cur_sum + action) >= 2:
            action = 0
        self.cur_sum += action
        return str(action) + '\n'
    
    def re_training(self):
        pass




            
        