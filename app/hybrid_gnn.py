import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GINConv, GATConv, GraphConv, global_mean_pool, global_max_pool

class HybridGNN(nn.Module):
    def __init__(self, hidden_channels, descriptor_size, dropout_rate):
        super(HybridGNN, self).__init__()

        self.conv1 = GINConv(Linear(9, hidden_channels))
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)

        self.gelu = nn.GELU()
        self.dropout = Dropout(p=dropout_rate)

        combined_size = hidden_channels * 2 + descriptor_size

        self.shared_fc = nn.Sequential(
            Linear(combined_size, 128),
            nn.GELU(),
            Dropout(dropout_rate)
        )

        self.tg_head = nn.Sequential(
            Linear(128, 64),
            nn.GELU(),
            Dropout(dropout_rate),
            Linear(64, 1)
        )

        self.tm_head = nn.Sequential(
            Linear(128, 64),
            nn.GELU(),
            Dropout(dropout_rate),
            Linear(64, 1)
        )

    def forward(self, x, edge_index, batch_index, descriptors):
        h = self.gelu(self.conv1(x, edge_index))
        h = self.gelu(self.conv2(h, edge_index))
        h = self.gelu(self.conv3(h, edge_index))

        h = torch.cat([
            global_max_pool(h, batch_index),
            global_mean_pool(h, batch_index)
        ], dim=1)

        combined = torch.cat([h, descriptors], dim=1)
        shared_features = self.shared_fc(combined)

        tg_out = self.tg_head(shared_features)
        tm_out = self.tm_head(shared_features)
        return torch.cat([tg_out, tm_out], dim=1), shared_features