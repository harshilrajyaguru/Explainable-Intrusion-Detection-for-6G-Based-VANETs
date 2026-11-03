import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLSTM(nn.Module):
    def __init__(self, input_size=37, hidden_size=128, num_layers=2, dropout=0.3, num_classes=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Binary head (logits for 2 classes: 0=Benign, 1=Malicious)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
        # Multi-class attackType head (4 classes: Benign, Blackhole, DDoS, Sybil)
        self.multi_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        # Last-step pooling: lstm_out[:, -1, :]
        last_step = lstm_out[:, -1, :]
        last_step = self.dropout(last_step)
        
        bin_logits = self.binary_head(last_step)
        multi_logits = self.multi_head(last_step)
        
        return bin_logits, multi_logits

class FocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ManyToManyLSTM(nn.Module):
    def __init__(self, input_size=37, hidden_size=128, num_layers=2, dropout=0.3, num_classes=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Binary head applied across all timesteps
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
        # Multi-class head applied across all timesteps
        self.multi_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x) # (batch_size, seq_len, hidden_size)
        lstm_out = self.dropout(lstm_out)
        
        bin_logits = self.binary_head(lstm_out)     # (batch_size, seq_len, 2)
        multi_logits = self.multi_head(lstm_out)   # (batch_size, seq_len, num_classes)
        
        return bin_logits, multi_logits

