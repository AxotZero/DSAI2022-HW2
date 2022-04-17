from pdb import set_trace as bp
import torch
import torch.nn as nn


class GruClassifier(nn.Module):
    def __init__(self, num_feat=4, hidden_size=32):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=num_feat,
            hidden_size=hidden_size,
            num_layers=3,
            dropout=0.3,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_size, 3)
    
    def forward(self, x, h0=None):
        if h0 is not None:
            embs, ht = self.gru(x, h0)
        else:
            embs, ht = self.gru(x)
        # bp()
        outs = self.classifier(embs)
        return outs, ht 
