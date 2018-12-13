
"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import constant, torch_utils

class AttentionNet(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # parameters
        self.hidden_dim = opt['hidden_dim']
        self.emb_dim = opt['emb_dim']
        
        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()
        
        # attn layer
        self.subj_attn_fc =  torch.nn.Sequential(
            torch.nn.Linear(600, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, 1)
        )

        self.obj_attn_fc =  torch.nn.Sequential(
            torch.nn.Linear(600, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, 1)
        )
        
        # output mlp layers
        in_dim = self.emb_dim*4
        layers = [nn.Linear(in_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def attention_net(self, prefs, supervisor, attn_fc):
        
        # attn_weights = torch.bmm(hiddens, tri.unsqueeze(2)).squeeze(2)
        attn_weights = torch.zeros(prefs.size(0), prefs.size(1)).to('cuda')
        for i in range(prefs.size(1)):
            tmp = torch.cat([prefs[:,i], supervisor], 1)
            attn_weights[:,i] = attn_fc(tmp).squeeze(1)

        soft_attn_weights = F.softmax(attn_weights, 1)
        attn_output = torch.bmm(prefs.transpose(1, 2),
                                soft_attn_weights.unsqueeze(2)).squeeze(2)
        return attn_output

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        word_emb = self.emb(words)
        # print(word_emb.shape)
        
        # pooling
        pool_type = self.opt['pooling']
        subj_mask = subj_pos.eq(0).eq(0).unsqueeze(2)
        obj_mask = obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        subj_emb = pool(word_emb, subj_mask, type=pool_type)
        obj_emb = pool(word_emb, obj_mask, type=pool_type)

        subj_attn = self.attention_net(word_emb, subj_emb, self.subj_attn_fc)
        obj_attn = self.attention_net(word_emb, subj_emb, self.subj_attn_fc)
        
        outputs = torch.cat([subj_attn, obj_attn, subj_emb, obj_emb], dim=1)
        outputs = self.out_mlp(outputs)
        
        return outputs

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

