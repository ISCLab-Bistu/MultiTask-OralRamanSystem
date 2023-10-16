# 定义一维空间注意力模块
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, input_dim, heads=1):
        super(SpatialAttention, self).__init__()
        self.heads = heads
        self.input_dim = input_dim

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        query = self.query(x).view(batch_size, seq_len, self.heads, self.input_dim // self.heads)
        key = self.key(x).view(batch_size, seq_len, self.heads, self.input_dim // self.heads)
        value = self.value(x).view(batch_size, seq_len, self.heads, self.input_dim // self.heads)

        query = query.transpose(1, 2).contiguous().view(batch_size * self.heads, seq_len, self.input_dim // self.heads)
        key = key.transpose(1, 2).contiguous().view(batch_size * self.heads, seq_len, self.input_dim // self.heads)
        value = value.transpose(1, 2).contiguous().view(batch_size * self.heads, seq_len, self.input_dim // self.heads)

        scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = self.softmax(scores)

        out = torch.bmm(attention_weights, value).view(batch_size, self.heads, seq_len, self.input_dim // self.heads)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.heads * (self.input_dim // self.heads))

        return out
