import torch
import torch.nn as nn

class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # use last timestep's hidden state (or map each timestep)
        preds = self.fc(out)  # reconstruct sequence
        return preds
