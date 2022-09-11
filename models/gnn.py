# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.init as init


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, text, adj):
        text = torch.matmul(adj.float(), text) / (torch.sum(adj, dim=-1, keepdim=True) + 1)
        text = self.w(text)
        return text


class GraphAttention(nn.Module):

    def __init__(self, in_features, out_features, leaky_relu=0.2):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, out_features)
        self.a_src = nn.Linear(out_features, 1, bias=False)
        self.a_dst = nn.Linear(out_features, 1, bias=False)
        self.activate = nn.LeakyReLU(leaky_relu)

    def forward(self, text, adj):
        hidden = self.w(text)  # B, L, out
        a_output = self.a_src(hidden).unsqueeze(2) + self.a_dst(hidden).unsqueeze(1)
        a_output = self.activate(a_output).squeeze(-1).masked_fill(mask=adj.bool(), value=-10000)
        attention = torch.softmax(a_output, dim=-1)  # B, L, L
        return torch.matmul(attention, hidden)


class AttentionGNN(nn.Module):

    def __init__(self, in_features, out_features):
        super(AttentionGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tempreture = math.sqrt(in_features)
        self.q = nn.Linear(in_features, out_features)
        self.k = nn.Linear(in_features, out_features)
        self.v = nn.Linear(in_features, out_features)

    def forward(self, text, adj):
        attention_score = torch.softmax(torch.matmul(self.q(text), self.k(text).transpose(1, 2)) / self.tempreture, dim=-1)
        new_adj = torch.matmul(attention_score, adj.float())
        output = torch.matmul(new_adj.float(), self.v(text))
        return output, new_adj
