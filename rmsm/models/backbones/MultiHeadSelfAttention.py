import math

import numpy as np
import torch
import torch.nn as nn


# # 定义位置编码器
# class PositionalEncoding(nn.Module):
#     '''
#     The positional encoding class is used in the encoder and decoder layers.
#     It's role is to inject sequence order information into the data since self-attention
#     mechanisms are permuatation equivariant. Naturally, this is not required in the static
#     transformer since there is no concept of 'order' in a portfolio.'''
#
#     def __init__(self, window, d_model):
#         super().__init__()
#
#         self.register_buffer('d_model', torch.tensor(d_model, dtype=torch.float64))
#
#         pe = torch.zeros(window, d_model)
#         for pos in range(window):
#             for i in range(0, d_model, 2):
#                 pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
#
#             for i in range(1, d_model, 2):
#                 pe[pos, i] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
#
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return x * torch.sqrt(self.d_model) + self.pe[:, :x.size(1)]


# 定义多头自注意力层
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "输入维度必须是头数的整数倍"

        # 定义权重矩阵
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)

        # 输出投影矩阵
        self.W_o = nn.Linear(input_dim, input_dim)

        # 位置编码层
        # self.positional_encoding = PositionalEncoding(900, 900)

    def forward(self, x):
        # x = self.positional_encoding(x)

        batch_size, seq_len, input_dim = x.size()

        # 使用线性映射来获取查询、键、值
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # 将查询、键、值分成多个头
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 计算多头注意力分数
        attention_scores = torch.einsum('bqhd,bkhd->bhqk', [queries, keys]) / (self.head_dim ** 0.5)

        # 使用softmax获取注意力权重
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 使用注意力权重对值进行加权求和
        weighted_values = torch.einsum('bhqk,bkhd->bqhd', [attention_weights, values])

        # 将多头的输出连接起来
        concatenated_values = weighted_values.view(batch_size, seq_len, -1)

        # 使用输出投影矩阵进行线性变换
        output = self.W_o(concatenated_values)

        return output
