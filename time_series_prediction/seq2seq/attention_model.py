import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by the number of heads"
        
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Split into multiple heads
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention weights
        energy = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention = self.softmax(energy)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Final linear layer
        out = self.fc_out(out)
        
        return out.mean(dim=1)


class ImprovedModel(nn.Module):
    def __init__(self, input_size, output_size, n=2):
        super().__init__()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, 128 * n, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.layer_norm1 = nn.LayerNorm(256 * n)
        
        self.lstm2 = nn.LSTM(256 * n, 128 * n, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.layer_norm2 = nn.LayerNorm(256 * n)
        
        self.lstm3 = nn.LSTM(256 * n, 64 * n, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(0.3)
        self.layer_norm3 = nn.LayerNorm(128 * n)
        
        # Attention mechanism
        self.attention = MultiHeadAttention(128 * n, 8 * n)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * n, 64 * n)
        self.fc2 = nn.Linear(64 * n, output_size)

    def forward(self, x):
        # Forward pass through LSTM layers with layer norm and dropout
        x, _ = self.lstm1(x)
        x = self.layer_norm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.layer_norm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.layer_norm3(x)
        x = self.dropout3(x)
        
        # Apply multi-head attention
        x = self.attention(x)
        
        # Forward pass through fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x