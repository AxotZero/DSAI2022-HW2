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
                 kfold=5,
                 epochs=400, 
                 lr=1e-2, 
                 model_hidden_size=16,
                 num_past=20):

        self.num_past = num_past
        self.kfold = kfold
        self.epochs = epochs
        self.model_hidden_size = model_hidden_size
        self.lr = lr
        self.models = []

    def train_one_fold(self, datas, targets, fold_idx=0, num_fold=5):
        print(f'=== train fold {fold_idx+1} ===')

        # compute valid size
        valid_ratio = 1 / num_fold if num_fold > 1 else 0.2
        valid_size = round(valid_ratio * datas.shape[0])

        # split train, valid
        idx_full = np.arange(len(datas))
        valid_idx = idx_full[fold_idx*valid_size: (fold_idx+1)*valid_size]
        first_idx, last_idx = valid_idx[0], valid_idx[-1]

        delete_idx = []
        delete_idx.append(valid_idx)
        if fold_idx != 0:
            delete_idx.append(np.array(range(valid_idx[0]-self.num_past, valid_idx[0])))
        if fold_idx != self.kfold-1:
            delete_idx.append(np.array(range(valid_idx[-1], valid_idx[-1] + self.num_past)))
        delete_idx = np.concatenate(delete_idx)
            
        train_idx = np.delete(idx_full, delete_idx)

        train_x, train_y = datas[train_idx], targets[train_idx]
        valid_x, valid_y = datas[valid_idx], targets[valid_idx]

        train_x = torch.tensor(train_x).float()
        train_y = torch.tensor(train_y).float()
        valid_x = torch.tensor(valid_x).float()
        valid_y = torch.tensor(valid_y).float()
        
        # training setup
        model = GruRatioRegressor(
            num_feat=train_x.size()[-1],
            num_out= 2,
            hidden_size=self.model_hidden_size,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr
        )
        criterion = nn.MSELoss()

        # core metric
        best_val_loss = float("inf")
        best_epoch = -1

        # start training
        pbar = tqdm(range(self.epochs), ncols=120)
        for epoch in pbar:

            # train
            model.train()
            pred, _ = model(train_x)
            train_loss = criterion(pred, train_y)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # valid
            model.eval()
            pred, _ = model(valid_x)
            valid_loss = criterion(pred, valid_y)

            pbar.set_description(f'epoch: {epoch}, train_mse: {train_loss.item():.6f}, valid_mse: {valid_loss.item():.6f}')
            pbar.update()

            # check whether get better result
            if valid_loss.item() < best_val_loss:
                best_val_loss = valid_loss.item()
                best_epoch = epoch
                torch.save(model.state_dict(), f"./data/model_data/model{fold_idx}.pth")
        print(f'fold: {fold_idx+1}, best_val_loss: {best_val_loss:.6f} at epoch {best_epoch}')

        return model



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


        for i in range(self.kfold):
            model = self.train_one_fold(datas, targets, fold_idx=i, num_fold=self.kfold)
            self.models.append(model)

    
    def predict_action(self, row):
        
        # first prediction
        if self.mode == TraderMode.TRAINING:
            print('=== start run test ===')

            self.mode = TraderMode.TESTING

            # load model
            for i, model in enumerate(self.models):
                model.load_state_dict(torch.load(f"./data/model_data/model{i}.pth"))
                model.eval()
            self.cur_sum = 0


        # append new row to raw_data
        row = np.array(row).reshape(1, -1)
        row = self.x_normalizer.transform(row)
        self.raw_data = np.concatenate((self.raw_data, row), 0)
        _input = self.raw_data[-self.num_past:]
        
        # predict
        preds = []
        with torch.no_grad():
            _input = torch.tensor(_input).unsqueeze(0).float()
            for model in self.models:
                pred, _ = model(_input)
                pred = pred[:, -1].detach().squeeze().numpy()
                preds.append(pred)
            pred = np.mean(np.array(preds), axis=1)


        # generate action
        action = BUY_IDX if pred[1] > pred[0] else SELL_IDX
        print(action, end=' ')
        # print(pred)
        if abs(self.cur_sum + action) >= 2:
            action = 0
        self.cur_sum += action

        return str(action) + '\n'
    
    def re_training(self):
        pass





            
        