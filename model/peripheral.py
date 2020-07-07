from enum import Enum
from random import random

import torch
from torch import nn


class AttnType(Enum):
    dot = 1
    general = 2
    concat = 3
    location = 4



class Attn(nn.Module):
    def __init__(self, attn_type: AttnType, hidden_dim, context_dim, mlp_dim=None):
        super(Attn, self).__init__()
        self.attn_type = attn_type
        if self.attn_type == AttnType.concat:
            assert mlp_dim is not None
            self.fh = nn.Linear(hidden_dim + context_dim, mlp_dim, bias=False)
            self.mlp_vec = nn.Parameter(torch.tensor([random()] * mlp_dim))
        if self.attn_type == AttnType.general:
            self.fh = nn.Linear(context_dim, hidden_dim, bias=False)
        if self.attn_type == AttnType.dot:
            self.fh = lambda x: x  # mimic identity layer
        if self.attn_type == AttnType.location:
            self.fh = nn.Linear(hidden_dim, 1)

    def forward(self, hidden, context):  # hidden:(H), context:(V,H)
        if self.attn_type == AttnType.concat:
            context_af = torch.tanh(self.fh(torch.cat((hidden.repeat(context.size(0), 1), context), dim=1)))
            score = torch.mm(context_af, self.mlp_vec.unsqueeze(1))
            # print("score.size:{}, context.size{}".format(score.size(), context.size()))
            score = torch.softmax(score.transpose(0, 1), dim=1)
            # print("score.size:{}, context.size{}".format(score.size(), context.size()))
            context = torch.mm(score, context)
            return context.squeeze(0)
        if self.attn_type == AttnType.general:
            # print("context, query size:{} {}".format(hidden.size(), (self.fh(context)).size()))
            score = torch.mm(self.fh(context), hidden.unsqueeze(1))
            score = torch.softmax(score.transpose(0, 1), dim=1)
            context = torch.mm(score, context)
            # print("context size:{}".format(context))
            return context.squeeze(0)  # torch.cat((context.squeeze(0), hidden), dim=0) #(H_E+H_D)
        raise ValueError("Undefined behavior")