import random
import json
import math

from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F

import pdb

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    

class LanguageMLP1(nn.Module):
    def __init__(self, num_language_features, nl_project, low_rank_dim, language_emb_dropout):
        super(LanguageMLP1, self).__init__()
        self.nonlinear_project = nn.Linear(num_language_features, nl_project)
        self.down_project = nn.Linear(nl_project, low_rank_dim) #32
        self.activation = F.relu
        self.dropout = nn.Dropout(language_emb_dropout) #0.1

    def forward(self, lang_vector, dev='cuda:0'):
        lang_emb = self.nonlinear_project(torch.tensor(lang_vector).to('cuda:0'))
        lang_emb = self.activation(lang_emb)
        lang_emb = self.down_project(lang_emb)
        lang_emb = self.dropout(lang_emb)
        return lang_emb

class LanguageMLP2(nn.Module):
    def __init__(self, num_language_features, nl_project, low_rank_dim, language_emb_dropout):
        super(LanguageMLP2, self).__init__()
        self.nonlinear_project = nn.Linear(num_language_features, nl_project)
        self.down_project = nn.Linear(nl_project, low_rank_dim) #32
        self.activation = gelu_new
        self.dropout = nn.Dropout(language_emb_dropout) #0.1

    def forward(self, lang_vector):

        #lang_vector = self._encode_language_ids(lang_ids, self.do_onehot)
        lang_emb = self.nonlinear_project(torch.tensor(lang_vector).to('cuda:0'))
        lang_emb = self.activation(lang_emb)
        lang_emb = self.down_project(lang_emb)
        lang_emb = self.dropout(lang_emb)
        return lang_emb

class LanguageMLP3(nn.Module):
    def __init__(self, num_language_features, nl_project, low_rank_dim, language_emb_dropout):
        super(LanguageMLP3, self).__init__()
        self.nonlinear_project = nn.Linear(num_language_features, low_rank_dim)
        #self.down_project = nn.Linear(nl_project, low_rank_dim) #32
        self.activation = F.relu
        self.dropout = nn.Dropout(language_emb_dropout) #0.1

    def forward(self, lang_vector, dev='cuda:0'):

        #lang_vector = self._encode_language_ids(lang_ids, self.do_onehot)
        lang_emb = self.nonlinear_project(lang_vector.clone().detach().requires_grad_(True).to('cuda:0'))
        lang_emb = self.activation(lang_emb)
        # lang_emb = self.down_project(lang_emb)
        # lang_emb = self.dropout(lang_emb)
        return lang_emb

class LanguageMLP4(nn.Module):
    def __init__(self, num_language_features, nl_project, low_rank_dim, language_emb_dropout):
        super(LanguageMLP4, self).__init__()
        self.nonlinear_project = nn.Linear(num_language_features, low_rank_dim)
        #self.down_project = nn.Linear(nl_project, low_rank_dim) #32
        self.activation = gelu_new
        self.dropout = nn.Dropout(language_emb_dropout) #0.1

    def forward(self, lang_vector):

        #lang_vector = self._encode_language_ids(lang_ids, self.do_onehot)
        lang_emb = self.nonlinear_project(torch.tensor(lang_vector).to('cuda:0'))
        lang_emb = self.activation(lang_emb)
        # lang_emb = self.down_project(lang_emb)
        # lang_emb = self.dropout(lang_emb)
        return lang_emb

class LanguageMLP5(nn.Module):
    def __init__(self, num_language_features, nl_project, low_rank_dim, language_emb_dropout):
        super(LanguageMLP5, self).__init__()
        self.nonlinear_project = nn.Linear(num_language_features, low_rank_dim)
        #self.down_project = nn.Linear(nl_project, low_rank_dim) #32
        self.activation = F.relu
        self.dropout = nn.Dropout(language_emb_dropout) #0.1

    def forward(self, lang_vector):

        #lang_vector = self._encode_language_ids(lang_ids, self.do_onehot)
        lang_emb = self.nonlinear_project(torch.tensor(lang_vector).to('cuda:0'))
        #lang_emb = self.activation(lang_emb)
        # lang_emb = self.down_project(lang_emb)
        # lang_emb = self.dropout(lang_emb)
        return lang_emb
