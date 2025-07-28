import torch.nn as nn
import torch

class LSTMEmotionClassifier(nn.Module):
    def __init__(self, input_dim=120, hidden_size=128, num_layers=2, num_classes=6):
        super(LSTMEmotionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def attention_net(self, lstm_output):
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)  # (batch, time, 1)
        context = torch.sum(attn_weights * lstm_output, dim=1)       # (batch, hidden*2)
        return context

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.attention_net(lstm_out)  
        out = self.fc1(out)                    
        out = self.bn1(out)                 
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
