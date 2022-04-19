from pdb import set_trace as bp

import pandas as pd
import numpy as np

from .constant import (
    COLS, CLASS_ACTION_MAP,
    NONE_CLASS_IDX, BUY_CLASS_IDX, SELL_CLASS_IDX,
    NONE_IDX, BUY_IDX, SELL_IDX
)
from .profit_calculator import calculate_profit, check_stock_actions_length


class MetricTracker:
    def __init__(self, num_pred=20):
        self.acc = 0
        self.profit = 0
        self.count = 0
        self.loss = 0
        self.num_pred = num_pred
    
    def compute_acc(self, pred, target):
        # pred = pred[:, -(self.num_pred):-1]
        pred = np.argmax(pred, axis=1)
        match = (pred == target)
        acc = np.mean(match)
        return acc

    def compute_profit(self, preds, target):
        # get stock df
        columns = COLS + list(range(target[0].shape[-1] - 4))
        stock_dfs = [
            pd.DataFrame(data=t, columns=columns)
            for t in target
        ]
        # get action
        actions = []
        preds = preds.reshape(len(stock_dfs), -1, 3)
        for pred in preds:
            pred = np.argmax(pred, axis=1)
            action = [CLASS_ACTION_MAP[p] for p in pred]
            cur_sum = 0
            for i in range(len(action)):
                if abs(cur_sum + action[i]) >= 2:
                    action[i] = 0
                cur_sum += action[i]
            actions.append(action)

        # get profit
        profits = []
        for stock_df, action in zip(stock_dfs, actions):
            if not check_stock_actions_length(stock_df, action):
                raise InvalidActionNumError('Invalid number of actions')
            profits.append(calculate_profit(stock_df, action))
            # bp()
        
        return np.mean(profits)

    def update(self, pred_y, class_y, raw_y, loss):
        pred_y = pred_y.detach().numpy()
        class_y = class_y.detach().numpy()
        # raw_y = raw_y.detach().numpy()
        count = len(pred_y)
        self.count += count
        self.acc += self.compute_acc(pred_y, class_y) * count
        self.profit += self.compute_profit(pred_y, raw_y) * count
        self.loss += loss * count
    
    def __str__(self):
        return f'loss: {self.loss/self.count: .3f}, acc: {self.acc/self.count :.3f}, profit: {self.profit/self.count : .3f}'

        



